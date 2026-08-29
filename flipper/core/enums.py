"""Enumerations for audio processing."""

from enum import Enum


class AudioFormat(Enum):
    """Supported audio file formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"

    def __str__(self) -> str:
        return self.value


class ProcessingMode(Enum):
    """Audio processing modes."""

    ANALYSIS = "analysis"
    ENHANCEMENT = "enhancement"
    SYNTHESIS = "synthesis"

    def __str__(self) -> str:
        return self.value
