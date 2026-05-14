# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file implements optional speaker verification for openWakeWord.
# It is a self-contained capability — like the VAD model in vad.py — that
# Model composes in only when the caller enables it. With speaker
# verification off, none of this code runs and none of its (heavy)
# dependencies are imported.
#
# The design mirrors openWakeWord's existing optional features:
#   - a standalone class (cf. vad.VAD) that wraps a model + does one job
#   - constructed only when enabled, branched on at runtime, zero cost off
#   - the caller owns enrollment + storage; openWakeWord stays
#     pure-inference and is handed pre-computed speaker embeddings
#
# What it does: given a short audio segment (the audio that triggered a
# wake word), produce a speaker embedding and cosine-match it against a
# catalogue of enrolled speakers, returning the best-matching speaker id
# (or "" when no enrolled speaker clears the similarity threshold).

# Imports
import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("openwakeword.speaker_verification")

# Speaker embedding dimension. The bundled CAM++ model emits 192-dim
# embeddings; _fit_to_dim defends against a future backend that doesn't.
EMBEDDING_DIM = 192

# Speaker verification operates on 16 kHz mono audio, matching the rest
# of openWakeWord.
SAMPLE_RATE = 16000


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    """Return v scaled to unit L2 norm. A near-zero vector is returned
    unchanged (normalising it would divide by ~0)."""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _fit_to_dim(emb: np.ndarray) -> np.ndarray:
    """Pad / truncate an embedding to EMBEDDING_DIM. The bundled CAM++
    backend natively emits EMBEDDING_DIM, so this is defence against a
    future backend that doesn't, not a routine code path."""
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    if emb.size == EMBEDDING_DIM:
        return emb
    logger.warning(
        "speaker-verification: backend emitted %d-dim embedding; "
        "padding/truncating to %d",
        emb.size, EMBEDDING_DIM,
    )
    out = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    n = min(emb.size, EMBEDDING_DIM)
    out[:n] = emb[:n]
    return out


def quality_score(samples: np.ndarray) -> float:
    """A simple RMS-energy quality estimate in [0, 1] for an audio
    segment: silent → 0, loud + steady → 1. Exposed so callers can gate
    on it (a near-silent segment makes a poor speaker embedding however
    well it happens to cosine-match)."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms <= 0.005:
        return 0.0
    if rms >= 0.2:
        return 1.0
    return (rms - 0.005) / (0.2 - 0.005)


def _to_float32(samples: np.ndarray) -> np.ndarray:
    """Normalise input audio to float32 [-1, 1]. Accepts int16 PCM (the
    format openWakeWord works in) or already-float audio."""
    if samples.dtype == np.float32 or samples.dtype == np.float64:
        return samples.astype(np.float32)
    # int16 (or other int) PCM → float32 [-1, 1]
    return samples.astype(np.float32) / 32768.0


class _SileroTrimmer:
    """Optional voice-activity trim: keep only the voice-active spans of
    a segment before embedding. A wake-driven capture is mostly silence
    around a short utterance; trimming to the speech produces a much
    stronger speaker embedding.

    Self-contained so SpeakerVerification does not depend on the
    Model's own VAD being enabled. Falls back to passthrough on any
    failure or when the segment is too short — better to embed the raw
    input than nothing."""

    def __init__(self, threshold: float = 0.5) -> None:
        # Local import: silero-vad is part of the speaker-verification
        # extra, only needed when this trimmer is actually constructed.
        from silero_vad import (  # type: ignore[import-untyped]
            load_silero_vad,
            get_speech_timestamps,
        )
        self._model = load_silero_vad(onnx=True)
        self._get_timestamps = get_speech_timestamps
        self._threshold = threshold
        logger.info("speaker-verification: silero-vad trimmer loaded")

    def trim(self, samples: np.ndarray) -> np.ndarray:
        if samples.size < SAMPLE_RATE // 2:  # < 0.5 s — too short to trim
            return samples
        try:
            import torch  # type: ignore[import-untyped]
            wav = torch.from_numpy(samples)
            timestamps = self._get_timestamps(
                wav,
                self._model,
                threshold=self._threshold,
                sampling_rate=SAMPLE_RATE,
                return_seconds=False,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "speaker-verification: silero-vad trim failed; passthrough",
                extra={"err": str(e)},
            )
            return samples
        if not timestamps:
            return samples
        chunks = [samples[t["start"]:t["end"]] for t in timestamps]
        return np.concatenate(chunks) if chunks else samples


class SpeakerVerification:
    """Optional speaker verification for openWakeWord.

    Given a short audio segment, produces a speaker embedding and
    cosine-matches it against a catalogue of *enrolled* speakers,
    returning the best-matching speaker id and its cosine score.

    openWakeWord stays pure-inference: enrollment (recording a speaker,
    computing their reference embedding, storing it) is the caller's
    responsibility. The caller hands SpeakerVerification a dict of
    pre-computed embeddings — `{speaker_id: [embedding, ...]}` — and
    SpeakerVerification only does the embed-and-match at detection time.

    The embedding backend is 3D-Speaker's CAM++ model
    (`iic/speech_campplus_sv_zh-cn_16k-common`), loaded via the
    modelscope SDK. modelscope + torch are in the optional
    `speaker-verification` extra; they are imported lazily here so a
    plain openWakeWord install never pulls them.

    Usage is normally indirect — via Model(speaker_verification=True,
    speaker_enrollments=...) — but the class can also be used
    standalone:

        sv = SpeakerVerification(enrollments={"alice": [emb]})
        speaker_id, score = sv.identify(audio_segment)
    """

    name = "campplus_sv"

    def __init__(
        self,
        enrollments: Optional[Dict[str, List[np.ndarray]]] = None,
        model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common",
        threshold: float = 0.65,
        vad_trim: bool = True,
        vad_threshold: float = 0.5,
    ) -> None:
        """Initialise the speaker-verification model.

        Args:
            enrollments (dict): The catalogue of enrolled speakers, as
                `{speaker_id: [embedding, ...]}`. Each embedding is a
                pre-computed EMBEDDING_DIM-length vector (a list or
                numpy array). A speaker may have several reference
                embeddings; identify() matches against all of them and
                keeps the best. May be empty / None and supplied later
                via `set_enrollments`.
            model_id (str): The modelscope model id for the CAM++
                speaker-verification model. The default is the 192-dim
                CAM++ base; the modelscope SDK fetches it on first
                construction and caches it.
            threshold (float): The cosine-similarity floor. identify()
                returns an empty speaker id when no enrolled speaker
                clears this.
            vad_trim (bool): Whether to trim the input segment to its
                voice-active spans before embedding (recommended — a
                wake capture is mostly silence). Requires silero-vad
                from the speaker-verification extra.
            vad_threshold (float): Speech-probability threshold for the
                VAD trimmer, when vad_trim is enabled.
        """
        # Local imports — modelscope + torch are only needed when
        # speaker verification is actually used. A plain openWakeWord
        # install does not have them and never reaches this line.
        from modelscope.pipelines import pipeline  # type: ignore[import-untyped]
        from modelscope.utils.constant import Tasks  # type: ignore[import-untyped]

        self._threshold = threshold
        self._model_id = model_id

        logger.info(
            "speaker-verification: loading CAM++ pipeline",
            extra={"model_id": model_id},
        )
        self._pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=model_id,
        )
        logger.info(
            "speaker-verification: CAM++ pipeline ready",
            extra={"model_id": model_id},
        )

        # Optional voice-activity trim before embedding.
        self._trimmer: Optional[_SileroTrimmer] = None
        if vad_trim:
            try:
                self._trimmer = _SileroTrimmer(threshold=vad_threshold)
            except Exception as e:  # pragma: no cover — defensive
                # A missing silero-vad install should degrade to
                # "embed the raw segment", not crash the whole model.
                logger.warning(
                    "speaker-verification: vad trimmer unavailable; "
                    "embedding will use the raw segment",
                    extra={"err": str(e)},
                )
                self._trimmer = None

        # The enrolled-speaker catalogue: speaker_id -> list of
        # L2-normalised reference embeddings. Held as a flat list of
        # (speaker_id, embedding) for a simple match loop.
        self._catalogue: List[Tuple[str, np.ndarray]] = []
        self.set_enrollments(enrollments or {})

    def set_enrollments(
        self, enrollments: Dict[str, List[np.ndarray]]
    ) -> int:
        """Replace the enrolled-speaker catalogue. Each value is a list
        of pre-computed embeddings for that speaker; they are
        L2-normalised on ingest so identify() can cosine-match with a
        plain dot product. Malformed entries are skipped with a warning.

        Returns the number of (speaker, embedding) pairs accepted."""
        catalogue: List[Tuple[str, np.ndarray]] = []
        for speaker_id, embeddings in enrollments.items():
            if not speaker_id:
                continue
            for emb in embeddings:
                arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                if arr.size != EMBEDDING_DIM:
                    logger.warning(
                        "speaker-verification: skipping enrollment with "
                        "unexpected embedding dim",
                        extra={
                            "speaker_id": speaker_id,
                            "got": arr.size,
                            "want": EMBEDDING_DIM,
                        },
                    )
                    continue
                catalogue.append((speaker_id, _l2_normalise(arr)))
        self._catalogue = catalogue
        logger.info(
            "speaker-verification: enrollments applied",
            extra={
                "speakers": len(set(s for s, _ in catalogue)),
                "embeddings": len(catalogue),
            },
        )
        return len(catalogue)

    @property
    def enrolled_speaker_ids(self) -> List[str]:
        """The distinct speaker ids currently in the catalogue."""
        seen: Deque[str] = deque()
        out: List[str] = []
        for speaker_id, _ in self._catalogue:
            if speaker_id not in seen:
                seen.append(speaker_id)
                out.append(speaker_id)
        return out

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Produce a speaker embedding for an audio segment.

        Accepts int16 PCM or float audio at SAMPLE_RATE; trims to
        voice-active spans first when the VAD trimmer is enabled.
        Returns an EMBEDDING_DIM-length float32 vector (not yet
        L2-normalised — identify() normalises before matching). On any
        backend failure, returns a zero vector rather than raising."""
        audio = _to_float32(np.asarray(samples))
        if self._trimmer is not None:
            audio = self._trimmer.trim(audio)

        try:
            # The modelscope speaker-verification pipeline expects a
            # *list* of waveforms (its public API compares waveform 1
            # to waveform 2). A bare ndarray makes it iterate the
            # samples; wrapping in a single-element list opts into the
            # embedding-only branch. output_emb=True returns the raw
            # embedding tensor alongside the textual output.
            result = self._pipeline([audio], output_emb=True)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "speaker-verification: embedding failed; "
                "returning zero embedding",
                extra={"err": str(e)},
            )
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        emb = None
        if isinstance(result, dict):
            # modelscope versions vary on the key; check the known ones.
            emb = result.get("embs")
            if emb is None:
                emb = result.get("outputs") or result.get("output_emb")
        else:
            emb = result
        if emb is None:
            logger.warning(
                "speaker-verification: pipeline returned no embedding"
            )
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        return _fit_to_dim(emb)

    def identify(self, samples: np.ndarray) -> Tuple[str, float, float]:
        """Identify the speaker of an audio segment.

        Embeds the segment and cosine-matches it against every enrolled
        reference embedding. Returns
        `(speaker_id, cosine_score, quality_score)`:

          - speaker_id: the best-matching enrolled speaker when its
            cosine clears the threshold, else "".
          - cosine_score: the best cosine similarity found (-1.0 when
            the catalogue is empty).
          - quality_score: an RMS-energy estimate in [0, 1] of the
            input segment — a low value flags a quiet/sparse segment
            that makes a weak basis for identification regardless of
            the cosine.

        This is the method Model.predict() calls after a wake word
        fires, on the audio that triggered it."""
        quality = quality_score(_to_float32(np.asarray(samples)))
        if not self._catalogue:
            return "", -1.0, quality

        emb = _l2_normalise(self.embed(samples))
        best_id = ""
        best_cosine = -1.0
        for speaker_id, ref in self._catalogue:
            cosine = float(np.dot(emb, ref))
            if cosine > best_cosine:
                best_cosine = cosine
                best_id = speaker_id

        if best_cosine < self._threshold:
            return "", best_cosine, quality
        return best_id, best_cosine, quality
