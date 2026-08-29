"""Abstract base classes for audio processing."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from flipper.core.interfaces import IAudioProcessor


class BaseProcessor(IAudioProcessor, ABC):
    """Base class for all audio processors."""

    def __init__(self, name: str):
        """Initialize the processor.

        Args:
            name: Human-readable name for this processor
        """
        self._name = name
        self._metadata: dict = {}

    def get_name(self) -> str:
        """Return the name of this processor."""
        return self._name

    @abstractmethod
    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio data and return the result.

        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Processed audio data
        """
        pass

    def set_metadata(self, key: str, value: any) -> None:
        """Store metadata about the processing."""
        self._metadata[key] = value

    def get_metadata(self) -> dict:
        """Retrieve metadata about the processing."""
        return self._metadata.copy()


class BaseSpectrogram(ABC):
    """Base class for spectrogram implementations."""

    def __init__(self, name: str = "Spectrogram"):
        """Initialize the spectrogram generator.

        Args:
            name: Name of the spectrogram method
        """
        self._name = name

    def get_name(self) -> str:
        """Return the name of this spectrogram method."""
        return self._name

    @abstractmethod
    def compute(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute spectrogram from audio data."""
        pass

    @abstractmethod
    def to_db(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert spectrogram to dB scale."""
        pass
