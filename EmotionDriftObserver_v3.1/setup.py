# EmotionDriftObserver_v3.1/setup.py

from setuptools import setup, find_packages

setup(
    name="hyper_diarizer",
    version="0.1.0",
    description="HyperDiazer speaker diarization module",
    packages=find_packages(include=["hyper_diarizer", "hyper_diarizer.*"]),
    install_requires=[
        "torch>=1.10.0",
        "torchaudio",
        "numpy",
        "pyannote.core>=4.0",      # or pin to 1.x if you prefer
        "pyannote.metrics",
        "speechbrain",
        "resemblyzer",
        # add any other deps your hyper-diarizer code needs
    ],
    entry_points={
        "console_scripts": [
            "hyperdiarizer=hyper_diarizer.hyper_diarizer:main",
        ]
    },
)
