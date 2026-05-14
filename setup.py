import platform
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Build extras_requires based on platform
def build_additional_requires():
    # py_version = platform.python_version()[0:3].replace('.', "")
    # if platform.system() == "Linux" and platform.machine() == "x86_64":
    #     additional_requires=[
    #         f"speexdsp_ns @ https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp{py_version}-cp{py_version}-linux_x86_64.whl",
    #     ]
    # elif platform.system() == "Linux" and platform.machine() == "aarch64":
    #     additional_requires=[
    #         f"speexdsp_ns @ https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp{py_version}-cp{py_version}-linux_aarch64.whl",
    #     ],
    if platform.system() == "Windows" and platform.machine() == "x86_64":
        additional_requires = [
            'PyAudioWPatch'
        ]
    else:
        additional_requires = []

    return additional_requires


setuptools.setup(
    name="openwakeword",
    version="0.6.0",
    # NOTE (xavros fork): `onnxruntime` was moved out of `install_requires` into the
    # `cpu` extra below. The stock library hard-pins `onnxruntime>=1.10.0,<2`, which
    # forces the CPU-only wheel to be installed even alongside `onnxruntime-gpu` (the
    # two share the `onnxruntime` import namespace and conflict). Downstreams that want
    # GPU inference install `onnxruntime-gpu` themselves; downstreams that want the
    # stock CPU behaviour install `openwakeword[cpu]`.
    install_requires=[
        'ai-edge-litert>=2.0.2,<3; platform_system == "Linux" or platform_system == "Darwin"',
        'speexdsp-ns>=0.1.2,<1; platform_system == "Linux"',
        'tqdm>=4.0,<5.0',
        'scipy>=1.3,<2',
        'scikit-learn>=1,<2',
        'requests>=2.0,<3',
    ],
    extras_require={
        'cpu': [
                    'onnxruntime>=1.10.0,<2',
                ],
        # Optional speaker verification (openwakeword.SpeakerVerification +
        # Model(speaker_verification=True)). The 3D-Speaker CAM++ backend is
        # distributed and run through modelscope; silero-vad provides the
        # optional pre-embed voice-activity trim. These are heavy deps
        # (torch), kept out of the base install — a plain openWakeWord user
        # who never enables speaker verification installs none of this.
        # modelscope needs the [framework] extra for the pipeline runtime.
        'speaker-verification': [
                    'modelscope[framework]>=1.18',
                    'torch>=2.0',
                    'silero-vad>=5.1',
                ],
        'test': [
                    'onnxruntime>=1.10.0,<2',
                    'pytest>=7.2.0,<8',
                    'pytest-cov>=2.10.1,<3',
                    'pytest-flake8>=1.1.1,<2',
                    'flake8>=5.0,<7.1',
                    'pytest-mypy>=0.10.0,<1',
                    'types-requests',
                    'types-PyYAML',
                    'mock>=5.1,<6',
                    'types-mock>=5.1,<6',
                    'types-requests>=2.0,<3'
                ],
        'full': [
                    'mutagen>=1.46.0,<2',
                    'torch>=1.13.1,<3',
                    'torchaudio>=0.13.1,<1',
                    'torchinfo>=1.8.0,<2',
                    'torchmetrics>=0.11.4,<1',
                    'speechbrain>=0.5.14,<1',
                    'audiomentations>=0.30.0,<1',
                    'torch-audiomentations>=0.11.0,<1',
                    'tqdm>=4.64.0,<5',
                    'pytest>=7.2.0,<8',
                    'pytest-cov>=2.10.1,<3',
                    'pytest-flake8>=1.1.1,<2',
                    'pytest-mypy>=0.10.0,<1',
                    'acoustics>=0.2.6,<1',
                    'pyyaml>=6.0,<7',
                    'tensorflow-cpu==2.8.1',
                    'tensorflow_probability==0.16.0',
                    'protobuf>=3.20,<4',
                    'onnx_tf==1.10.0',
                    'onnx==1.14.0',
                    'pronouncing>=0.2.0,<1',
                    'datasets>=2.14.4,<3',
                    'deep-phonemizer==0.0.19'
                ]
    },
    author="David Scripka",
    author_email="david.scripka@gmail.com",
    description="An open-source audio wake word (or phrase) detection framework with a focus on performance and simplicity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/openwakeword",
    project_urls={
        "Bug Tracker": "https://pypi.org/project/openwakeword/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)