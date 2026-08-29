"""Interfaces and protocols for audio processing."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class IAudioProcessor(ABC):
    """Interface for audio processors."""

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

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this processor."""
        pass


class ISpectrogram(ABC):
    """Interface for spectrogram generation."""

    @abstractmethod
    def compute(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute spectrogram from audio data.

        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Spectrogram matrix
        """
        pass

    @abstractmethod
    def to_db(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert spectrogram to dB scale."""
        pass
