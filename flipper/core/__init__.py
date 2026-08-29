"""Core abstractions and interfaces for audio processing."""

from flipper.core.abstracts import BaseProcessor
from flipper.core.enums import AudioFormat
from flipper.core.interfaces import IAudioProcessor

__all__ = ["BaseProcessor", "AudioFormat", "IAudioProcessor"]
