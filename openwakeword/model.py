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

# Imports
import numpy as np
import openwakeword
from openwakeword.utils import AudioFeatures, re_arg

import wave
import os
import logging
import functools
import pickle
from collections import deque, defaultdict
from dataclasses import dataclass, field
from functools import partial
import time
from typing import List, Union, DefaultDict, Dict


@dataclass
class PredictionResult:
    """Structured result returned by Model.predict() when speaker
    verification is enabled (or when return_result_object=True is
    passed explicitly).

    With speaker verification *disabled* and return_result_object not
    set, predict() returns the bare `scores` dict exactly as it always
    has — existing openWakeWord code is unaffected. This object is the
    richer shape for callers that have opted into speaker verification
    and need to know *who* spoke, not just *that* a wake word fired.

    Attributes:
        scores: The per-model wake-word scores in [0, 1] — identical to
            the dict predict() returns in its classic form.
        speaker_id: The id of the matched enrolled speaker, or "" when
            no wake word fired this frame, no speaker cleared the
            similarity threshold, or speaker verification is off.
        speaker_score: The cosine similarity of the best speaker match.
            -1.0 when speaker verification did not run (no wake word
            fired) or the enrollment catalogue is empty.
        speaker_quality: An RMS-energy quality estimate in [0, 1] of the
            audio segment the speaker embedding was computed from. 0.0
            when speaker verification did not run. A low value means a
            quiet/sparse segment — a weak basis for identification even
            if the cosine match looks high.
    """

    scores: Dict[str, float] = field(default_factory=dict)
    speaker_id: str = ""
    speaker_score: float = -1.0
    speaker_quality: float = 0.0


# Define main model class
class Model():
    """
    The main model class for openWakeWord. Creates a model object with the shared audio pre-processer
    and for arbitrarily many custom wake word/wake phrase models.
    """
    @re_arg({"wakeword_model_paths": "wakeword_models"})  # temporary handling of keyword argument change
    def __init__(
            self,
            wakeword_models: List[str] = [],
            class_mapping_dicts: List[dict] = [],
            enable_speex_noise_suppression: bool = False,
            vad_threshold: float = 0,
            custom_verifier_models: dict = {},
            custom_verifier_threshold: float = 0.1,
            speaker_verification: bool = False,
            speaker_enrollments: dict = {},
            speaker_verification_threshold: float = 0.65,
            speaker_verification_model: str = "campplus_sv",
            inference_framework: str = "tflite",
            device: str = "cpu",
            **kwargs
            ):
        """Initialize the openWakeWord model object.

        Args:
            wakeword_models (List[str]): A list of paths of ONNX/tflite models to load into the openWakeWord model object.
                                              If not provided, will load all of the pre-trained models. Alternatively,
                                              just the names of pre-trained models can be provided to select a subset of models.
            class_mapping_dicts (List[dict]): A list of dictionaries with integer to string class mappings for
                                              each model in the `wakeword_models` arguments
                                              (e.g., {"0": "class_1", "1": "class_2"})
            enable_speex_noise_suppression (bool): Whether to use the noise suppresion from the SpeexDSP
                                                   library to pre-process all incoming audio. May increase
                                                   model performance when reasonably stationary background noise
                                                   is present in the environment where openWakeWord will be used.
                                                   It is very lightweight, so enabling it doesn't significantly
                                                   impact efficiency.
            vad_threshold (float): Whether to use a voice activity detection model (VAD) from Silero
                                   (https://github.com/snakers4/silero-vad) to filter predictions.
                                   For every input audio frame, a VAD score is obtained and only those model predictions
                                   with VAD scores above the threshold will be returned. The default value (0),
                                   disables voice activity detection entirely.
            custom_verifier_models (dict): A dictionary of paths to custom verifier models, where
                                           the keys are the model names (corresponding to the openwakeword.MODELS
                                           attribute) and the values are the filepaths of the
                                           custom verifier models.
            custom_verifier_threshold (float): The score threshold to use a custom verifier model. If the score
                                               from a model for a given frame is greater than this value, the
                                               associated custom verifier model will also predict on that frame, and
                                               the verifier score will be returned.
            speaker_verification (bool): Whether to enable optional speaker verification — identifying *which*
                                         enrolled speaker spoke the wake word, not just *that* one was spoken.
                                         When True, after a wake word fires, the audio that triggered it is run
                                         through a speaker-embedding model and cosine-matched against the
                                         enrolled-speaker catalogue; predict() then returns a PredictionResult
                                         (carrying .speaker_id / .speaker_score) instead of the bare scores
                                         dict. Disabled by default — when off, none of the speaker-verification
                                         code runs and its (heavy) modelscope/torch dependencies are never
                                         imported. Requires the `speaker-verification` extra.
            speaker_enrollments (dict): The enrolled-speaker catalogue, as
                                        {speaker_id: [embedding, ...]}. Each embedding is a pre-computed
                                        speaker-embedding vector — openWakeWord stays pure-inference, so
                                        enrollment (recording a speaker, computing their reference embedding,
                                        storing it) is the caller's responsibility. Only used when
                                        speaker_verification is True. Can also be supplied/replaced later via
                                        the SpeakerVerification.set_enrollments method on
                                        model.speaker_verification.
            speaker_verification_threshold (float): The cosine-similarity floor for a speaker match. Below this,
                                                    the speaker is reported as unidentified ("").
            speaker_verification_model (str): Which entry in openwakeword.SPEAKER_MODELS to use as the
                                              speaker-embedding backend. Default "campplus_sv" (3D-Speaker
                                              CAM++, 192-dim).
            inference_framework (str): The inference framework to use when for model prediction. Options are
                                       "tflite" or "onnx". The default is "tflite" as this results in better
                                       efficiency on common platforms (x86, ARM64), but in some deployment
                                       scenarios ONNX models may be preferable.
            device (str): The device to run inference on, either "cpu" or "gpu" (default "cpu"). When "gpu"
                          and the onnx inference framework is selected, every ONNX session (the wakeword
                          models, the melspectrogram + embedding preprocessor models, and the VAD model)
                          is created with the CUDAExecutionProvider, falling back to CPUExecutionProvider
                          if CUDA is unavailable at runtime. Requires the `onnxruntime-gpu` package. Has
                          no effect with the tflite inference framework.
            kwargs (dict): Any other keyword arguments to pass the the preprocessor instance
        """
        # Get model paths for pre-trained models if user doesn't provide models to load
        pretrained_model_paths = openwakeword.get_pretrained_model_paths(inference_framework)
        wakeword_model_names = []
        if wakeword_models == []:
            wakeword_models = pretrained_model_paths
            wakeword_model_names = list(openwakeword.MODELS.keys())
        elif len(wakeword_models) >= 1:
            for ndx, i in enumerate(wakeword_models):
                if os.path.exists(i):
                    wakeword_model_names.append(os.path.splitext(os.path.basename(i))[0])
                else:
                    # Find pre-trained path by modelname
                    matching_model = [j for j in pretrained_model_paths if i.replace(" ", "_") in j.split(os.path.sep)[-1]]
                    if matching_model == []:
                        raise ValueError("Could not find pretrained model for model name '{}'".format(i))
                    else:
                        wakeword_models[ndx] = matching_model[0]
                        wakeword_model_names.append(i)

        # Create attributes to store models and metadata
        self.models = {}
        self.model_inputs = {}
        self.model_outputs = {}
        self.model_prediction_function = {}
        self.class_mapping = {}
        self.custom_verifier_models = {}
        self.custom_verifier_threshold = custom_verifier_threshold

        # Do imports for  inference framework
        if inference_framework == "tflite":
            try:
                import ai_edge_litert.interpreter as tflite

                def tflite_predict(tflite_interpreter, input_index, output_index, x):
                    tflite_interpreter.set_tensor(input_index, x)
                    tflite_interpreter.invoke()
                    return tflite_interpreter.get_tensor(output_index)[None, ]

            except ImportError:
                logging.warning("Tried to import the tflite runtime, but it was not found. "
                                "Trying to switching to onnxruntime instead, if appropriate models are available.")
                if wakeword_models != [] and all(['.onnx' in i for i in wakeword_models]):
                    inference_framework = "onnx"
                elif wakeword_models != [] and all([os.path.exists(i.replace('.tflite', '.onnx')) for i in wakeword_models]):
                    inference_framework = "onnx"
                    wakeword_models = [i.replace('.tflite', '.onnx') for i in wakeword_models]
                else:
                    raise ValueError("Tried to import the LiteRT runtime for provided LiteRT models, but it was not found. "
                                     "Please install it using `pip install ai-edge-litert`")

        if inference_framework == "onnx":
            try:
                import onnxruntime as ort

                def onnx_predict(onnx_model, x):
                    return onnx_model.run(None, {onnx_model.get_inputs()[0].name: x})

            except ImportError:
                raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

        for mdl_path, mdl_name in zip(wakeword_models, wakeword_model_names):
            # Load openwakeword models
            if inference_framework == "onnx":
                if ".tflite" in mdl_path:
                    raise ValueError("The onnx inference framework is selected, but tflite models were provided!")

                sessionOptions = ort.SessionOptions()
                sessionOptions.inter_op_num_threads = 1
                sessionOptions.intra_op_num_threads = 1

                # When device == "gpu", prefer CUDA but keep CPU in the list as an
                # explicit fallback so a host without a working CUDA provider still
                # loads instead of raising.
                wakeword_providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                                      if device == "gpu" else ["CPUExecutionProvider"])
                self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions,
                                                             providers=wakeword_providers)

                self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
                self.model_outputs[mdl_name] = self.models[mdl_name].get_outputs()[0].shape[1]
                pred_function = functools.partial(onnx_predict, self.models[mdl_name])
                self.model_prediction_function[mdl_name] = pred_function

            if inference_framework == "tflite":
                if ".onnx" in mdl_path:
                    raise ValueError("The tflite inference framework is selected, but onnx models were provided!")

                self.models[mdl_name] = tflite.Interpreter(model_path=mdl_path, num_threads=1)
                self.models[mdl_name].allocate_tensors()

                self.model_inputs[mdl_name] = self.models[mdl_name].get_input_details()[0]['shape'][1]
                self.model_outputs[mdl_name] = self.models[mdl_name].get_output_details()[0]['shape'][1]

                tflite_input_index = self.models[mdl_name].get_input_details()[0]['index']
                tflite_output_index = self.models[mdl_name].get_output_details()[0]['index']

                pred_function = functools.partial(tflite_predict, self.models[mdl_name], tflite_input_index, tflite_output_index)
                self.model_prediction_function[mdl_name] = pred_function

            if class_mapping_dicts and class_mapping_dicts[wakeword_models.index(mdl_path)].get(mdl_name, None):
                self.class_mapping[mdl_name] = class_mapping_dicts[wakeword_models.index(mdl_path)]
            elif openwakeword.model_class_mappings.get(mdl_name, None):
                self.class_mapping[mdl_name] = openwakeword.model_class_mappings[mdl_name]
            else:
                self.class_mapping[mdl_name] = {str(i): str(i) for i in range(0, self.model_outputs[mdl_name])}

            # Load custom verifier models
            if isinstance(custom_verifier_models, dict):
                if custom_verifier_models.get(mdl_name, False):
                    self.custom_verifier_models[mdl_name] = pickle.load(open(custom_verifier_models[mdl_name], 'rb'))

            if len(self.custom_verifier_models.keys()) < len(custom_verifier_models.keys()):
                raise ValueError(
                    "Custom verifier models were provided, but some were not matched with a base model!"
                    " Make sure that the keys provided in the `custom_verifier_models` dictionary argument"
                    " exactly match that of the `.models` attribute of an instantiated openWakeWord Model object"
                    " that has the same base models but doesn't have custom verifier models."
                )

        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(partial(deque, maxlen=30))

        # Initialize SpeexDSP noise canceller
        if enable_speex_noise_suppression:
            from speexdsp_ns import NoiseSuppression
            self.speex_ns = NoiseSuppression.create(160, 16000)
        else:
            self.speex_ns = None

        # Initialize Silero VAD
        self.vad_threshold = vad_threshold
        if vad_threshold > 0:
            self.vad = openwakeword.VAD(device=device)

        # Initialize optional speaker verification. Like the VAD above,
        # this is constructed only when enabled — when disabled,
        # self.speaker_verification is None, predict() never branches
        # into the speaker path, and the (heavy) modelscope/torch
        # dependencies are never imported.
        self.speaker_verification_enabled = bool(speaker_verification)
        self.speaker_verification = None
        if self.speaker_verification_enabled:
            if openwakeword.SpeakerVerification is None:
                raise ImportError(
                    "speaker_verification=True was requested, but the speaker "
                    "verification dependencies are not installed. Install the "
                    "`speaker-verification` extra: pip install "
                    "openwakeword[speaker-verification]"
                )
            if speaker_verification_model not in openwakeword.SPEAKER_MODELS:
                raise ValueError(
                    f"unknown speaker_verification_model "
                    f"{speaker_verification_model!r}; available: "
                    f"{sorted(openwakeword.SPEAKER_MODELS)}"
                )
            modelscope_id = openwakeword.SPEAKER_MODELS[
                speaker_verification_model
            ]["modelscope_id"]
            self.speaker_verification = openwakeword.SpeakerVerification(
                enrollments=speaker_enrollments,
                model_id=modelscope_id,
                threshold=speaker_verification_threshold,
            )

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(inference_framework=inference_framework, device=device, **kwargs)

    def get_parent_model_from_label(self, label):
        """Gets the parent model associated with a given prediction label"""
        parent_model = ""
        for mdl in self.class_mapping.keys():
            if label in self.class_mapping[mdl].values():
                parent_model = mdl
            elif label in self.class_mapping.keys() and label == mdl:
                parent_model = mdl

        return parent_model

    def get_providers(self):
        """Report the execution provider(s) actually bound by every loaded ONNX session.

        Useful for confirming GPU placement: a session created with the
        CUDAExecutionProvider can silently fall back to the CPUExecutionProvider when
        the CUDA libraries are missing or incompatible, and the only reliable signal is
        what onnxruntime reports back after session creation.

        Returns:
            dict: A mapping of session name -> list of provider strings. Keys include
                  each wakeword model name, "melspectrogram", "embedding", and "vad"
                  (the last only when voice activity detection is enabled). Returns an
                  empty dict for the tflite inference framework, which has no concept of
                  onnxruntime execution providers.
        """
        providers = {}
        for mdl_name, session in self.models.items():
            if hasattr(session, "get_providers"):
                providers[mdl_name] = session.get_providers()
        if hasattr(self.preprocessor.melspec_model, "get_providers"):
            providers["melspectrogram"] = self.preprocessor.melspec_model.get_providers()
        if hasattr(self.preprocessor.embedding_model, "get_providers"):
            providers["embedding"] = self.preprocessor.embedding_model.get_providers()
        if self.vad_threshold > 0 and hasattr(self.vad.model, "get_providers"):
            providers["vad"] = self.vad.model.get_providers()
        return providers

    def reset(self):
        """Reset the prediction and audio feature buffers. Useful for re-initializing the model, though may not be efficient
        when called too frequently."""
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))
        self.preprocessor.reset()

    def predict(self, x: np.ndarray, patience: dict = {},
                threshold: dict = {}, debounce_time: float = 0.0, timing: bool = False,
                speaker_verification_seconds: float = 3.0,
                return_result_object: bool = False):
        """Predict with all of the wakeword models on the input audio frames

        Args:
            x (ndarray): The input audio data to predict on with the models. Ideally should be multiples of 80 ms
                                (1280 samples), with longer lengths reducing overall CPU usage
                                but decreasing detection latency. Input audio with durations greater than or less
                                than 80 ms is also supported, though this will add a detection delay of up to 80 ms
                                as the appropriate number of samples are accumulated.
            patience (dict): How many consecutive frames (of 1280 samples or 80 ms) above the threshold that must
                             be observed before the current frame will be returned as non-zero.
                             Must be provided as an a dictionary where the keys are the
                             model names and the values are the number of frames. Can reduce false-positive
                             detections at the cost of a lower true-positive rate.
                             By default, this behavior is disabled.
            threshold (dict): The threshold values to use when the `patience` or `debounce_time` behavior is enabled.
                              Must be provided as an a dictionary where the keys are the
                              model names and the values are the thresholds.
            debounce_time (float): The time (in seconds) to wait before returning another non-zero prediction
                                   after a non-zero prediction. Can preven multiple detections of the same wake-word.
            timing (bool): Whether to return timing information of the models. Can be useful to debug and
                           assess how efficiently models are running on the current hardware.
            speaker_verification_seconds (float): How many seconds of recent audio to run speaker verification
                                                  over when a wake word fires (only used when speaker
                                                  verification is enabled). The audio is pulled from the
                                                  preprocessor's rolling buffer — i.e. the very audio the wake
                                                  word fired on — so no re-capture is needed.
            return_result_object (bool): Force predict() to return a PredictionResult object even when speaker
                                         verification is disabled. By default predict() returns the classic
                                         bare scores dict (or (dict, timing) tuple) when speaker verification
                                         is off, for backwards compatibility; set this True to always get the
                                         structured object.

        Returns:
            By default, a dict of scores between 0 and 1 for each model, where 0 indicates no
            wake-word/wake-phrase detected (and, if `timing` is true, a (scores, timing) tuple).

            When speaker verification is enabled (or `return_result_object` is true), returns a
            PredictionResult object instead: `.scores` is the same per-model score dict, and
            `.speaker_id` / `.speaker_score` / `.speaker_quality` carry the speaker-verification
            result for the wake word that fired this frame (empty / -1.0 / 0.0 when no wake word
            fired or no enrolled speaker matched). If `timing` is also true, returns a
            (PredictionResult, timing) tuple.
        """
        # Check input data type
        if not isinstance(x, np.ndarray):
            raise ValueError(f"The input audio data (x) must by a Numpy array, instead received an object of type {type(x)}.")

        # Setup timing dict
        if timing:
            timing_dict: Dict[str, Dict] = {}
            timing_dict["models"] = {}
            feature_start = time.time()

        # Get audio features (optionally with Speex noise suppression)
        if self.speex_ns:
            n_prepared_samples = self.preprocessor(self._suppress_noise_with_speex(x))
        else:
            n_prepared_samples = self.preprocessor(x)

        if timing:
            timing_dict["models"]["preprocessor"] = time.time() - feature_start

        # Get predictions from model(s)
        predictions = {}
        for mdl in self.models.keys():
            if timing:
                model_start = time.time()

            # Run model to get predictions
            if n_prepared_samples > 1280:
                group_predictions = []
                for i in np.arange(n_prepared_samples//1280-1, -1, -1):
                    group_predictions.extend(
                        self.model_prediction_function[mdl](
                            self.preprocessor.get_features(
                                    self.model_inputs[mdl],
                                    start_ndx=-self.model_inputs[mdl] - i
                            )
                        )
                    )
                prediction = np.array(group_predictions).max(axis=0)[None, ]
            elif n_prepared_samples == 1280:
                prediction = self.model_prediction_function[mdl](
                    self.preprocessor.get_features(self.model_inputs[mdl])
                )
            elif n_prepared_samples < 1280:  # get previous prediction if there aren't enough samples
                if self.model_outputs[mdl] == 1:
                    if len(self.prediction_buffer[mdl]) > 0:
                        prediction = [[[self.prediction_buffer[mdl][-1]]]]
                    else:
                        prediction = [[[0]]]
                elif self.model_outputs[mdl] != 1:
                    n_classes = max([int(i) for i in self.class_mapping[mdl].keys()])
                    prediction = [[[0]*(n_classes+1)]]

            if self.model_outputs[mdl] == 1:
                predictions[mdl] = prediction[0][0][0]
            else:
                for int_label, cls in self.class_mapping[mdl].items():
                    predictions[cls] = prediction[0][0][int(int_label)]

            # Update scores based on custom verifier model
            if self.custom_verifier_models != {}:
                for cls in predictions.keys():
                    if predictions[cls] >= self.custom_verifier_threshold:
                        parent_model = self.get_parent_model_from_label(cls)
                        if self.custom_verifier_models.get(parent_model, False):
                            verifier_prediction = self.custom_verifier_models[parent_model].predict_proba(
                                self.preprocessor.get_features(self.model_inputs[mdl])
                            )[0][-1]
                            predictions[cls] = verifier_prediction

            # Zero predictions for first 5 frames during model initialization
            for cls in predictions.keys():
                if len(self.prediction_buffer[cls]) < 5:
                    predictions[cls] = 0.0

            # Get timing information
            if timing:
                timing_dict["models"][mdl] = time.time() - model_start

        # Update scores based on thresholds or patience arguments
        if patience != {} or debounce_time > 0:
            if threshold == {}:
                raise ValueError("Error! When using the `patience` argument, threshold "
                                 "values must be provided via the `threshold` argument!")
            if patience != {} and debounce_time > 0:
                raise ValueError("Error! The `patience` and `debounce_time` arguments cannot be used together!")
            for mdl in predictions.keys():
                parent_model = self.get_parent_model_from_label(mdl)
                if predictions[mdl] != 0.0:
                    if parent_model in patience.keys():
                        scores = np.array(self.prediction_buffer[mdl])[-patience[parent_model]:]
                        if (scores >= threshold[parent_model]).sum() < patience[parent_model]:
                            predictions[mdl] = 0.0
                    elif debounce_time > 0:
                        if parent_model in threshold.keys():
                            n_frames = int(np.ceil(debounce_time/(n_prepared_samples/16000)))
                            recent_predictions = np.array(self.prediction_buffer[mdl])[-n_frames:]
                            if predictions[mdl] >= threshold[parent_model] and \
                               (recent_predictions >= threshold[parent_model]).sum() > 0:
                                predictions[mdl] = 0.0

        # Update prediction buffer
        for mdl in predictions.keys():
            self.prediction_buffer[mdl].append(predictions[mdl])

        # (optionally) get voice activity detection scores and update model scores
        if self.vad_threshold > 0:
            if timing:
                vad_start = time.time()

            self.vad(x)

            if timing:
                timing_dict["models"]["vad"] = time.time() - vad_start

            # Get frames from last 0.4 to 0.56 seconds (3 frames) before the current
            # frame and get max VAD score
            vad_frames = list(self.vad.prediction_buffer)[-7:-4]
            vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0
            for mdl in predictions.keys():
                if vad_max_score < self.vad_threshold:
                    predictions[mdl] = 0.0

        # Optional speaker verification. Runs only when enabled AND a
        # wake word actually fired this frame — so the (relatively
        # expensive) speaker-embedding inference is paid once per
        # detection, not on every 80 ms frame. The audio it runs over
        # is pulled from the preprocessor's own rolling buffer via
        # get_raw_audio(): that is the exact audio the wake word fired
        # on, including the wake word and the utterance onset, so there
        # is no separate capture and no detection-latency gap.
        speaker_id = ""
        speaker_score = -1.0
        speaker_quality = 0.0
        if self.speaker_verification is not None:
            if timing:
                sv_start = time.time()
            wake_fired = any(score >= 0.5 for score in predictions.values())
            if wake_fired:
                recent_audio = self.preprocessor.get_raw_audio(
                    n_seconds=speaker_verification_seconds
                )
                if recent_audio.size > 0:
                    speaker_id, speaker_score, speaker_quality = \
                        self.speaker_verification.identify(recent_audio)
            if timing:
                timing_dict["models"]["speaker_verification"] = time.time() - sv_start

        # Build the return value. Backwards compatible: when speaker
        # verification is off and return_result_object was not set,
        # predict() returns exactly what it always has — the bare
        # scores dict, or a (scores, timing) tuple. When speaker
        # verification is on (or the object is explicitly requested),
        # the richer PredictionResult is returned instead.
        if self.speaker_verification is not None or return_result_object:
            result = PredictionResult(
                scores=predictions,
                speaker_id=speaker_id,
                speaker_score=speaker_score,
                speaker_quality=speaker_quality,
            )
            if timing:
                return result, timing_dict
            return result

        if timing:
            return predictions, timing_dict
        else:
            return predictions

    def predict_clip(self, clip: Union[str, np.ndarray], padding: int = 1, chunk_size=1280, **kwargs):
        """Predict on an full audio clip, simulating streaming prediction.
        The input clip must bit a 16-bit, 16 khz, single-channel WAV file.

        Args:
            clip (Union[str, np.ndarray]): The path to a 16-bit PCM, 16 khz, single-channel WAV file,
                                           or an 1D array containing the same type of data
            padding (int): How many seconds of silence to pad the start/end of the clip with
                            to make sure that short clips can be processed correctly (default: 1)
            chunk_size (int): The size (in samples) of each chunk of audio to pass to the model
            kwargs: Any keyword arguments to pass to the class `predict` method

        Returns:
            list: A list containing the frame-level prediction dictionaries for the audio clip
        """
        if isinstance(clip, str):
            # Load audio clip as 16-bit PCM data
            with wave.open(clip, mode='rb') as f:
                # Load WAV clip frames
                data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        elif isinstance(clip, np.ndarray):
            data = clip

        if padding:
            data = np.concatenate(
                (
                    np.zeros(16000*padding).astype(np.int16),
                    data,
                    np.zeros(16000*padding).astype(np.int16)
                )
            )

        # Iterate through clip, getting predictions
        predictions = []
        step_size = chunk_size
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions.append(self.predict(data[i:i+step_size], **kwargs))

        return predictions

    def _get_positive_prediction_frames(
            self,
            file: str,
            threshold: float = 0.5,
            return_type: str = "features",
            **kwargs
            ):
        """
        Gets predictions for the input audio data, and returns the audio features (embeddings)
        or audio data for all of the frames with a score above the `threshold` argument.
        Can be a useful way to collect false-positive predictions.

        Args:
            file (str): The path to a 16-bit 16khz WAV audio file to process
            threshold (float): The minimum score required for a frame of audio features
                               to be returned.
            return_type (str): The type of data to return when a positive prediction is
                               detected. Can be either 'features' or 'audio' to return
                               audio embeddings or raw audio data, respectively.
            kwargs: Any keyword arguments to pass to the class `predict` method

        Returns:
            dict: A dictionary with filenames as keys and  N x M arrays as values,
                  where N is the number of examples and M is the number
                  of audio features, depending on the model input shape.
        """
        # Load audio clip as 16-bit PCM data
        with wave.open(file, mode='rb') as f:
            # Load WAV clip frames
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

        # Iterate through clip, getting predictions
        positive_data = defaultdict(list)
        step_size = 1280
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions = self.predict(data[i:i+step_size], **kwargs)
            for lbl in predictions.keys():
                if predictions[lbl] >= threshold:
                    mdl = self.get_parent_model_from_label(lbl)
                    features = self.preprocessor.get_features(self.model_inputs[mdl])
                    if return_type == 'features':
                        positive_data[lbl].append(features)
                    if return_type == 'audio':
                        context = data[max(0, i - 16000*3):i + 16000]
                        if len(context) == 16000*4:
                            positive_data[lbl].append(context)

        positive_data_combined = {}
        for lbl in positive_data.keys():
            positive_data_combined[lbl] = np.vstack(positive_data[lbl])

        return positive_data_combined

    def _suppress_noise_with_speex(self, x: np.ndarray, frame_size: int = 160):
        """
        Runs the input audio through the SpeexDSP noise suppression algorithm.
        Note that this function updates the state of the existing Speex noise
        suppression object, and isn't intended to be called externally.

        Args:
            x (ndarray): The 16-bit, 16khz audio to process. Must always be an
                         integer multiple of `frame_size`.
            frame_size (int): The frame size to use for the Speex Noise suppressor.
                              Must match the frame size specified during the
                              initialization of the noise suppressor.

        Returns:
            ndarray: The input audio with noise suppression applied
        """
        cleaned = []
        for i in range(0, x.shape[0], frame_size):
            chunk = x[i:i+frame_size]
            cleaned.append(self.speex_ns.process(chunk.tobytes()))

        cleaned_bytestring = b''.join(cleaned)
        cleaned_array = np.frombuffer(cleaned_bytestring, np.int16)
        return cleaned_array
