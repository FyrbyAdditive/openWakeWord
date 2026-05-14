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
