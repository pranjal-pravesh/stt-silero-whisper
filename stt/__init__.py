"""
Speech-to-Text package.
Provides functionality for capturing, buffering, and transcribing speech.
"""

from .speech_buffer import SpeechBuffer
from .transcription_worker import TranscriptionWorker

__all__ = ["SpeechBuffer", "TranscriptionWorker"] 