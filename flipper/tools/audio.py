"""Audio tools that wrap existing Flipper domain code."""

from typing import Any

import numpy as np

from flipper.core.abstracts import BaseProcessor
from flipper.core.enums import AudioFormat
from flipper.models.audio import AudioData
from flipper.services.processor import AudioProcessingService
from flipper.tools.base import BaseTool


class IdentityProcessor(BaseProcessor):
    """Pass-through processor used to prove the tool -> service path."""

    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        self.set_metadata("sample_rate", sample_rate)
        self.set_metadata("samples", int(audio_data.size))
        return audio_data


class ListAudioFormatsTool(BaseTool):
    """List formats Flipper knows about."""

    name = "list_audio_formats"
    description = "List supported audio file formats."
    parameters: dict[str, Any] = {}

    def run(self, **kwargs: Any) -> str:
        formats = ", ".join(fmt.value for fmt in AudioFormat)
        return f"Supported formats: {formats}"


class ProcessAudioTool(BaseTool):
    """Run a short silent clip through AudioProcessingService."""

    name = "process_audio"
    description = (
        "Generate a short silent clip and run it through the identity "
        "audio processing pipeline."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "duration_seconds": {"type": "number", "default": 0.1},
            "sample_rate": {"type": "integer", "default": 22050},
        },
    }

    def run(
        self,
        duration_seconds: float = 0.1,
        sample_rate: int = 22050,
        **kwargs: Any,
    ) -> str:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        n_samples = max(1, int(duration_seconds * sample_rate))
        waveform = np.zeros(n_samples, dtype=np.float32)
        duration = n_samples / float(sample_rate)
        audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            duration=duration,
            channels=1,
            metadata={"source": "process_audio_tool"},
        )

        service = AudioProcessingService()
        service.add_processor(IdentityProcessor("identity"))
        processed = service.process(audio)

        return (
            f"Processed {processed.duration:.3f}s of audio at "
            f"{processed.sample_rate} Hz through identity processor "
            f"(shape={processed.get_shape()})."
        )
