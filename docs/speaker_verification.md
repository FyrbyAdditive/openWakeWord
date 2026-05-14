# Speaker Verification

Speaker verification is an optional openWakeWord capability that identifies *which*
enrolled speaker spoke a wake word — turning a wake event into a wake event *plus an
identity*. It is distinct from [custom verifier models](custom_verifier_models.md):
a custom verifier model is a yes/no filter ("was this likely one of my users?"),
whereas speaker verification returns a specific speaker id.

It is disabled by default. When disabled, none of the speaker-verification code runs
and none of its dependencies are imported — a plain openWakeWord install is entirely
unaffected.

# When to use it

Speaker verification is useful when an application needs to act differently per person
*directly from the wake word* — personalised responses, per-user permissions,
multi-user devices — without bolting on a separate identification step after the wake
event.

As with custom verifier models, there are trade-offs:

1. You need to be able to *enroll* every intended user — record some of their speech
   and compute a reference embedding. openWakeWord does not do enrollment for you (see
   below); it is a small, one-time effort per user.

2. Speaker verification is a similarity match against enrolled references. Accuracy
   depends on the enrollment audio being reasonably representative of the deployment
   acoustic environment, and on the utterance being long/clear enough to embed well.

# Design

Speaker verification is built as a self-contained capability — like the VAD model —
that `Model` composes in only when enabled. The embedding backend is
[3D-Speaker](https://github.com/modelscope/3D-Speaker)'s CAM++ model
(`iic/speech_campplus_sv_zh-cn_16k-common`, 192-dim embeddings), loaded via the
modelscope SDK.

openWakeWord stays a **pure-inference** library: *enrollment* — recording a speaker,
computing their reference embedding, and storing it — is the application's
responsibility. You pass `Model` a catalogue of *pre-computed* speaker embeddings, and
openWakeWord does only the embed-and-match step, when a wake word fires.

Crucially, the speaker embedding is computed from the audio that **already triggered
the wake word** — openWakeWord's preprocessor keeps a rolling buffer of recent raw
audio, and the speaker step reads the most recent few seconds out of it. There is no
second capture and no gap between "wake word fired" and "audio for identification" —
it is the same audio.

The speaker step runs *only when a wake word actually fired* on a given frame, so the
(relatively expensive) speaker-embedding inference is paid once per detection, not on
every audio frame.

# Installation

Speaker verification needs heavier dependencies (the modelscope SDK + torch) than the
openWakeWord base install. They live in an optional extra:

```
pip install openwakeword[speaker-verification]
```

# Enrollment

openWakeWord expects pre-computed speaker embeddings. The embeddings must come from the
same model openWakeWord uses for verification — the CAM++ model above. The simplest
way to produce one is to run the modelscope CAM++ pipeline over a few seconds of a
speaker's speech and keep the returned embedding. The
`openwakeword.speaker_verification.SpeakerVerification` class can also be used
directly — its `embed()` method takes an audio segment and returns the embedding —
so an enrollment routine can reuse the exact same code path:

```python
from openwakeword.speaker_verification import SpeakerVerification

sv = SpeakerVerification()                       # no enrollments yet
alice_embedding = sv.embed(alice_audio_segment)  # 16 kHz mono PCM
```

Store these embeddings however your application prefers (a file, a database, etc.).
A speaker may have several reference embeddings — verification matches against all of
them and keeps the best.

# Usage

Construct the `Model` with `speaker_verification=True` and a `speaker_enrollments`
catalogue mapping `speaker_id` to a list of pre-computed embeddings:

```python
import openwakeword

owwModel = openwakeword.Model(
    wakeword_models=["hey_jarvis"],
    speaker_verification=True,
    speaker_enrollments={
        "alice": [alice_embedding],
        "bob":   [bob_embedding_1, bob_embedding_2],
    },
    speaker_verification_threshold=0.65,   # cosine-similarity floor for a match
)

result = owwModel.predict(audio_frame)
```

When speaker verification is enabled, `predict()` returns a `PredictionResult` object
instead of the bare scores dict:

- `result.scores` — the per-model wake-word scores in `[0, 1]`. This is exactly the
  dict `predict()` returns in its classic form.
- `result.speaker_id` — the id of the matched enrolled speaker, or `""` when no wake
  word fired this frame, or no enrolled speaker cleared the similarity threshold.
- `result.speaker_score` — the cosine similarity of the best speaker match (`-1.0`
  when speaker verification did not run, or the catalogue is empty).
- `result.speaker_quality` — an RMS-energy quality estimate in `[0, 1]` of the audio
  segment the speaker embedding was computed from. A low value means a quiet/sparse
  segment — a weak basis for identification even if the cosine match looks high.

With speaker verification **disabled**, `predict()` returns the classic scores `dict`
(or `(dict, timing)` tuple) unchanged — existing code is not affected. If you want the
`PredictionResult` object without enabling speaker verification, pass
`return_result_object=True` to `predict()`.

The enrollment catalogue can also be updated at runtime — for example when a new user
is enrolled — via the `SpeakerVerification` instance on the model:

```python
owwModel.speaker_verification.set_enrollments(updated_enrollments)
```
