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

# Tests for the optional speaker-verification capability
# (openwakeword/speaker_verification.py).
#
# Speaker verification's real backend (3D-Speaker CAM++) ships in the
# `speaker-verification` extra (modelscope + torch + silero-vad), which
# is NOT installed for the base `test` extra that CI runs. So these
# tests inject lightweight fakes for `modelscope` and `silero_vad` into
# sys.modules *before* importing SpeakerVerification — the lazy imports
# inside the class then resolve to the fakes. This exercises the real
# logic that matters (enrollment ingest + L2-normalisation, the cosine
# match loop, the similarity threshold, the result shape, the
# defensive paths) without needing the heavy ML stack.

import sys
import types

import numpy as np
import pytest

EMBEDDING_DIM = 192


# --- Fake modelscope + silero_vad, installed before the import -------

def _install_fakes(embed_vector: np.ndarray) -> None:
    """Install fake `modelscope` + `silero_vad` modules so
    `import openwakeword.speaker_verification` and constructing
    SpeakerVerification work without the real extra. `embed_vector` is
    what the fake CAM++ pipeline returns for any input."""

    # --- fake modelscope ---
    modelscope = types.ModuleType("modelscope")
    pipelines = types.ModuleType("modelscope.pipelines")
    utils = types.ModuleType("modelscope.utils")
    constant = types.ModuleType("modelscope.utils.constant")
    hub = types.ModuleType("modelscope.hub")
    snapshot = types.ModuleType("modelscope.hub.snapshot_download")

    class _Tasks:
        speaker_verification = "speaker-verification"

    def _pipeline(task=None, model=None, **kwargs):
        # Return a callable that mimics the CAM++ embedding pipeline:
        # given [audio], output_emb=True, return {"embs": <(1, D)>}.
        def _run(inputs, output_emb=False, **kw):
            return {"embs": np.asarray(embed_vector, dtype=np.float32).reshape(1, -1)}
        return _run

    def _snapshot_download(model_id, **kwargs):
        return f"/fake/modelscope/cache/{model_id}"

    # setattr (rather than `module.attr = ...`) so mypy does not flag
    # the dynamic attributes on these stub ModuleType instances.
    setattr(pipelines, "pipeline", _pipeline)
    setattr(constant, "Tasks", _Tasks)
    setattr(snapshot, "snapshot_download", _snapshot_download)
    setattr(utils, "constant", constant)
    setattr(hub, "snapshot_download", snapshot)
    setattr(modelscope, "pipelines", pipelines)
    setattr(modelscope, "utils", utils)
    setattr(modelscope, "hub", hub)

    sys.modules["modelscope"] = modelscope
    sys.modules["modelscope.pipelines"] = pipelines
    sys.modules["modelscope.utils"] = utils
    sys.modules["modelscope.utils.constant"] = constant
    sys.modules["modelscope.hub"] = hub
    sys.modules["modelscope.hub.snapshot_download"] = snapshot

    # --- fake silero_vad: passthrough trimmer (no speech-span cuts) ---
    silero = types.ModuleType("silero_vad")

    def _load_silero_vad(onnx=False):
        return object()

    def _get_speech_timestamps(wav, model, **kwargs):
        # Report the whole clip as one voice-active span.
        n = len(wav)
        return [{"start": 0, "end": n}] if n else []

    setattr(silero, "load_silero_vad", _load_silero_vad)
    setattr(silero, "get_speech_timestamps", _get_speech_timestamps)
    sys.modules["silero_vad"] = silero

    # Only fake `torch` when the real package is not installed. The
    # fake must carry a `Tensor` class: scipy's array_api_compat probes
    # `getattr(sys.modules["torch"], "Tensor")` to detect torch arrays
    # (scipy is pulled in transitively via sklearn ->
    # custom_verifier_model -> openwakeword.__init__), and a torch
    # stub without `Tensor` makes that probe raise. The fake's
    # `from_numpy` returns a plain ndarray, which is all the
    # passthrough silero trimmer needs.
    try:
        import torch  # noqa: F401  (real torch — use it as-is)
    except ImportError:
        torch_stub = types.ModuleType("torch")
        setattr(torch_stub, "Tensor", type("Tensor", (), {}))
        setattr(torch_stub, "from_numpy", lambda a: a)
        sys.modules["torch"] = torch_stub


def _make_sv(embed_vector, **kwargs):
    """Install the fakes for `embed_vector`, (re)import the module, and
    return a fresh SpeakerVerification."""
    _install_fakes(embed_vector)
    # Drop any cached import so the fakes are picked up.
    sys.modules.pop("openwakeword.speaker_verification", None)
    from openwakeword.speaker_verification import SpeakerVerification
    return SpeakerVerification(**kwargs)


def _emb(seed: int) -> np.ndarray:
    """A deterministic EMBEDDING_DIM-length embedding."""
    rng = np.random.RandomState(seed)
    return rng.randn(EMBEDDING_DIM).astype(np.float32)


def _pcm(seconds: float = 2.0, amplitude: int = 6000) -> np.ndarray:
    """N seconds of int16 PCM (a 220 Hz tone) at 16 kHz."""
    n = int(16000 * seconds)
    t = np.arange(n)
    return (amplitude * np.sin(2 * np.pi * 220 * t / 16000)).astype(np.int16)


# --- Tests -----------------------------------------------------------

def test_module_imports_and_exposes_class():
    """With the fakes installed, the module imports and the class is
    constructable. openwakeword/__init__.py also exports it."""
    sv = _make_sv(_emb(1))
    assert sv is not None
    assert sv.name == "campplus_sv"


def test_enrollment_ingest_and_speaker_ids():
    """set_enrollments accepts {speaker_id: [embedding, ...]} and
    reports the distinct speaker ids; a speaker may have several
    reference embeddings."""
    sv = _make_sv(_emb(1))
    n = sv.set_enrollments({
        "alice": [_emb(10)],
        "bob": [_emb(20), _emb(21)],
    })
    assert n == 3  # three (speaker, embedding) pairs
    assert set(sv.enrolled_speaker_ids) == {"alice", "bob"}


def test_enrollment_skips_bad_embedding_dim():
    """An embedding of the wrong length is skipped, not fatal."""
    sv = _make_sv(_emb(1))
    n = sv.set_enrollments({
        "alice": [_emb(10)],                       # good
        "bob": [np.zeros(7, dtype=np.float32)],    # wrong dim — skipped
    })
    assert n == 1
    assert sv.enrolled_speaker_ids == ["alice"]


def test_identify_hit():
    """When the segment embeds to (close to) an enrolled speaker's
    reference, identify() returns that speaker id and a cosine that
    clears the threshold. The fake pipeline returns a fixed vector, so
    we enrol that exact vector for 'alice'."""
    target = _emb(42)
    sv = _make_sv(target, threshold=0.65, vad_trim=False)
    sv.set_enrollments({"alice": [target], "bob": [_emb(99)]})
    speaker_id, score, quality = sv.identify(_pcm())
    assert speaker_id == "alice"
    assert score == pytest.approx(1.0, abs=1e-5)  # same vector → cosine 1
    assert 0.0 <= quality <= 1.0


def test_identify_miss_below_threshold():
    """When no enrolled speaker's embedding is close enough, identify()
    returns an empty speaker id but still reports the best cosine + the
    quality."""
    sv = _make_sv(_emb(1), threshold=0.99, vad_trim=False)
    # Enrol unrelated vectors — the fake embed (_emb(1)) won't hit 0.99
    # cosine against either.
    sv.set_enrollments({"alice": [_emb(2)], "bob": [_emb(3)]})
    speaker_id, score, quality = sv.identify(_pcm())
    assert speaker_id == ""
    assert score < 0.99
    assert 0.0 <= quality <= 1.0


def test_identify_empty_catalogue():
    """With no enrolled speakers, identify() returns ('', -1.0, quality)
    — it does not raise."""
    sv = _make_sv(_emb(1), vad_trim=False)
    speaker_id, score, quality = sv.identify(_pcm())
    assert speaker_id == ""
    assert score == -1.0
    assert 0.0 <= quality <= 1.0


def test_quality_score_silent_vs_loud():
    """quality_score reflects RMS energy: a silent segment scores 0, a
    loud steady tone scores high."""
    _install_fakes(_emb(1))
    sys.modules.pop("openwakeword.speaker_verification", None)
    from openwakeword.speaker_verification import quality_score
    silent = np.zeros(16000, dtype=np.int16)
    loud = _pcm(amplitude=20000)
    assert quality_score(silent) == 0.0
    assert quality_score(loud) > 0.5


def test_embed_returns_correct_dim():
    """embed() returns an EMBEDDING_DIM float32 vector."""
    sv = _make_sv(_emb(7), vad_trim=False)
    emb = sv.embed(_pcm())
    assert emb.shape == (EMBEDDING_DIM,)
    assert emb.dtype == np.float32


def test_set_enrollments_replaces_not_merges():
    """set_enrollments replaces the catalogue wholesale."""
    sv = _make_sv(_emb(1))
    sv.set_enrollments({"alice": [_emb(10)]})
    assert sv.enrolled_speaker_ids == ["alice"]
    sv.set_enrollments({"bob": [_emb(20)]})
    assert sv.enrolled_speaker_ids == ["bob"]  # alice gone, not merged
