import os
from openwakeword.model import Model
from openwakeword.vad import VAD
from openwakeword.custom_verifier_model import train_custom_verifier

# Speaker verification is an optional capability — its model backend
# (3D-Speaker CAM++) and dependencies (modelscope, torch) live in the
# `speaker-verification` extra. The import is guarded so a plain
# openWakeWord install (without the extra) still imports cleanly;
# `SpeakerVerification` is then None and Model raises a clear error if
# speaker verification is requested without the extra installed.
try:
    from openwakeword.speaker_verification import SpeakerVerification
    _HAS_SPEAKER_VERIFICATION = True
except ImportError:
    SpeakerVerification = None  # type: ignore[assignment,misc]
    _HAS_SPEAKER_VERIFICATION = False

__all__ = ['Model', 'VAD', 'train_custom_verifier', 'SpeakerVerification']

FEATURE_MODELS = {
    "embedding": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/embedding_model.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"
    },
    "melspectrogram": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/melspectrogram.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"
    }
}

VAD_MODELS = {
    "silero_vad": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/silero_vad.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx"
    }
}

# Speaker-verification model registry. Unlike MODELS / VAD_MODELS /
# FEATURE_MODELS — which are ONNX/tflite files fetched from GitHub
# release assets — the speaker-verification backend is a 3D-Speaker
# CAM++ model distributed through modelscope. The modelscope SDK
# fetches and caches it on first construction of SpeakerVerification,
# so the registry records the `modelscope_id` rather than a
# download_url. download_models() warms the modelscope cache when a
# speaker model name is requested.
SPEAKER_MODELS = {
    "campplus_sv": {
        "modelscope_id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "embedding_dim": 192,
    }
}

MODELS = {
    "alexa": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/alexa_v0.1.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.tflite"
    },
    "hey_mycroft": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_mycroft_v0.1.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_mycroft_v0.1.tflite"
    },
    "hey_jarvis": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_jarvis_v0.1.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite"
    },
    "hey_rhasspy": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_rhasspy_v0.1.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_rhasspy_v0.1.tflite"
    },
    "timer": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/timer_v0.1.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/timer_v0.1.tflite"
    },
    "weather": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/weather_v0.1.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/weather_v0.1.tflite"
    }
}

model_class_mappings = {
    "timer": {
        "1": "1_minute_timer",
        "2": "5_minute_timer",
        "3": "10_minute_timer",
        "4": "20_minute_timer",
        "5": "30_minute_timer",
        "6": "1_hour_timer"
    }
}


def get_pretrained_model_paths(inference_framework="tflite"):
    if inference_framework == "tflite":
        return [MODELS[i]["model_path"] for i in MODELS.keys()]
    elif inference_framework == "onnx":
        return [MODELS[i]["model_path"].replace(".tflite", ".onnx") for i in MODELS.keys()]
