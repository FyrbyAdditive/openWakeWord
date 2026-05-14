# xavros fork of openWakeWord

This is a fork of [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord),
maintained for the [xavros](https://github.com/FyrbyAdditive) household-AI project.
The upstream library is Apache-2.0 licensed; that license is retained unchanged (see
`LICENSE`).

## Why a fork

Upstream openWakeWord is effectively unmaintained — the last release (v0.6.0) was
February 2024, with 100+ open issues and a dozen stale PRs. The change xavros needs
(GPU execution provider support, see below) has no realistic path upstream, so this
fork is permanent. The fork's `main` branch tracks upstream untouched; all xavros
changes live on the `xavros` branch, so a future rebase onto a newer upstream stays
possible.

xavros pins this fork by **commit SHA**, never by branch.

## Changes on the `xavros` branch

All changes extend openWakeWord's *own existing* `device` convention (the
`AudioFeatures` class already accepted `device='cpu'|'gpu'`) rather than inventing new
API — keeping the diff minimal and rebaseable.

1. **`model.py` — `Model.__init__` accepts `device: str = "cpu"`.** Upstream's `Model`
   never accepted `device` and never passed it to its `AudioFeatures(...)`
   construction, so the preprocessor (melspectrogram + embedding) ONNX sessions were
   stuck on CPU even though `AudioFeatures` could already run on GPU. The wakeword
   model's own `ort.InferenceSession` was also hardcoded to `CPUExecutionProvider`.
   The fork threads `device` into the wakeword session, the `AudioFeatures(...)`
   construction, and the `VAD(...)` construction — so `Model(device="gpu")` moves the
   *entire* ONNX pipeline to CUDA, not just one session.

2. **`vad.py` — `VAD.__init__` accepts `device: str = "cpu"`.** The Silero VAD session
   was hardcoded to `CPUExecutionProvider` with no `device` parameter at all.

3. **`utils.py` / `vad.py` / `model.py` — fail-safe provider ordering.** When
   `device == "gpu"`, every session is now created with
   `["CUDAExecutionProvider", "CPUExecutionProvider"]` rather than CUDA alone, so a
   host without a working CUDA provider degrades to CPU instead of raising. Matters for
   an always-on household service.

4. **`model.py` — new `Model.get_providers()` accessor.** Returns a
   `{session_name: [provider, ...]}` map for every loaded ONNX session (wakeword
   models, `melspectrogram`, `embedding`, and `vad` when enabled). onnxruntime can
   silently fall back from CUDA to CPU; this is the reliable way for a downstream to
   confirm where inference actually landed.

5. **`setup.py` — `onnxruntime` moved out of `install_requires` into a `cpu` extra.**
   Upstream hard-pinned `onnxruntime>=1.10.0,<2`, which forces the CPU-only wheel even
   alongside `onnxruntime-gpu` (they share the `onnxruntime` import namespace and
   conflict). Downstreams wanting GPU install `onnxruntime-gpu` themselves; downstreams
   wanting stock CPU behaviour install `openwakeword[cpu]`. The `test` extra carries
   `onnxruntime` so the fork's own test suite still installs standalone.

## Changes on the `xavros-part3-consolidation` branch

Branched off `xavros`. Adds **optional speaker verification** as a first-class
openWakeWord capability — identifying *which* enrolled speaker spoke the wake word,
not just *that* one was spoken. This lets a single service do wake detection *and*
speaker identification off one audio stream (the xavros voice-wake-service consumes
it this way), and is generically useful to any openWakeWord user. It is fully
optional: disabled by default, with none of its (heavy) dependencies imported when
off — a plain openWakeWord install is unaffected. The design follows the library's
own `VAD` pattern (a standalone optional sub-capability `Model` composes in).

6. **`speaker_verification.py` (new) — `SpeakerVerification` class.** A self-contained
   speaker-embedding + cosine-match capability, modelled on `vad.py`. Backend is
   3D-Speaker's CAM++ model via the modelscope SDK; an optional silero-vad pre-embed
   trim is built in. openWakeWord stays pure-inference — the caller supplies a
   catalogue of *pre-computed* speaker embeddings (`{speaker_id: [embedding, ...]}`)
   and the class does the embed-and-match.

7. **`__init__.py` — `SPEAKER_MODELS` registry + guarded export.** A registry entry
   for the CAM++ model (recording the modelscope id, since it is not a GitHub release
   asset like the wakeword/VAD models). `SpeakerVerification` is exported but its
   import is guarded — a plain install without the extra still imports cleanly, and
   `Model` raises a clear error if speaker verification is requested without it.

8. **`utils.py` — `AudioFeatures.get_raw_audio(n_seconds)` accessor.** openWakeWord
   already keeps a rolling buffer of recent raw PCM (used internally for the streaming
   melspectrogram); this exposes it. Generic and useful on its own — it lets any
   caller retrieve the audio a wake word fired on without re-capturing it. Speaker
   verification uses it so the speaker embedding is computed from the exact audio that
   triggered the wake word.

9. **`utils.py` — `download_models()` warms the modelscope cache** for any requested
   speaker model; skipped (and modelscope never imported) unless a speaker model is
   explicitly requested.

10. **`model.py` — `Model` gains optional speaker-verification params + a structured
    result.** `Model.__init__` takes `speaker_verification` / `speaker_enrollments` /
    `speaker_verification_threshold` / `speaker_verification_model`, wired exactly like
    the existing optional VAD (constructed only when enabled, `None` otherwise, zero
    cost when off). A new `PredictionResult` dataclass carries `.scores` plus
    `.speaker_id` / `.speaker_score` / `.speaker_quality`. `predict()` runs the speaker
    step only when a wake word actually fired, on the recent buffered audio, and
    returns a `PredictionResult` when speaker verification is enabled — **but returns
    the classic bare scores dict unchanged when it is off**, so existing code is
    unaffected (`return_result_object=True` opts into the object regardless).

11. **`setup.py` — `speaker-verification` extra.** `modelscope[framework]`, `torch`,
    `silero-vad` — the heavy deps the feature needs, kept out of the base install.

Docs: `docs/speaker_verification.md` (mirrors `docs/custom_verifier_models.md`), a
README section, `tests/test_speaker_verification.py` (modelscope/silero stubbed so it
runs without the extra), and `examples/detect_with_speaker_verification.py`.
