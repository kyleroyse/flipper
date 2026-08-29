"""Audio data models and classes."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AudioData:
    """Represents audio data with metadata.

    Attributes:
        waveform: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        channels: Number of audio channels
        metadata: Additional metadata dictionary
    """

    waveform: np.ndarray
    sample_rate: int
    duration: float
    channels: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate audio data."""
        if self.waveform.size == 0:
            raise ValueError("Waveform cannot be empty")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")

    def get_shape(self) -> tuple:
        """Return the shape of the waveform."""
        return self.waveform.shape

    def to_mono(self) -> "AudioData":
        """Convert to mono by averaging channels."""
        if self.channels == 1:
            return AudioData(
                waveform=self.waveform.copy(),
                sample_rate=self.sample_rate,
                duration=self.duration,
                channels=1,
                metadata=self.metadata.copy(),
            )

        mono_waveform = np.mean(self.waveform, axis=0)
        return AudioData(
            waveform=mono_waveform,
            sample_rate=self.sample_rate,
            duration=self.duration,
            channels=1,
            metadata=self.metadata.copy(),
        )


@dataclass
class SpectrogramData:
    """Represents spectrogram data.

    Attributes:
        spectrogram: Spectrogram matrix
        frequencies: Frequency bins
        times: Time bins
        sample_rate: Original sample rate
    """

    spectrogram: np.ndarray
    frequencies: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None

    def get_shape(self) -> tuple:
        """Return the shape of the spectrogram."""
        return self.spectrogram.shape
