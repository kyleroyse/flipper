"""Audio processing service."""

from typing import List

import numpy as np

from flipper.core.interfaces import IAudioProcessor
from flipper.models.audio import AudioData


class AudioProcessingService:
    """Service for managing audio processing pipelines."""

    def __init__(self):
        """Initialize the audio processing service."""
        self._processors: List[IAudioProcessor] = []

    def add_processor(self, processor: IAudioProcessor) -> "AudioProcessingService":
        """Add a processor to the pipeline.

        Args:
            processor: Audio processor to add

        Returns:
            Self for chaining
        """
        self._processors.append(processor)
        return self

    def process(self, audio_data: AudioData) -> AudioData:
        """Process audio through all registered processors.

        Args:
            audio_data: Audio data to process

        Returns:
            Processed audio data
        """
        processed_waveform = audio_data.waveform.copy()

        for processor in self._processors:
            processed_waveform = processor.process(
                processed_waveform, audio_data.sample_rate
            )

        return AudioData(
            waveform=processed_waveform,
            sample_rate=audio_data.sample_rate,
            duration=audio_data.duration,
            channels=audio_data.channels,
            metadata=audio_data.metadata.copy(),
        )

    def get_processors(self) -> List[IAudioProcessor]:
        """Get all registered processors."""
        return self._processors.copy()

    def clear_processors(self) -> None:
        """Remove all processors from the pipeline."""
        self._processors.clear()
