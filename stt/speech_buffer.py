#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech buffer module for managing continuous audio streams.
Provides functionality to maintain a rolling buffer of audio chunks
and collect speech segments when VAD detects speech.
"""

import collections
from typing import Optional, Deque
from rich.console import Console

# Create a console for Rich output
console = Console()

class SpeechBuffer:
    """
    Maintains a rolling buffer of audio data with speech detection.
    
    This class keeps a pre-speech buffer (to capture the beginning of utterances)
    and a speech buffer (to collect the current speech segment).
    
    When speech ends, it returns the complete segment including pre-speech audio.
    """
    
    def __init__(self, pre_speech_ms: int = 500, sample_rate: int = 16000, 
                 bytes_per_sample: int = 2, channels: int = 1, verbose: bool = False):
        """
        Initialize the speech buffer.
        
        Args:
            pre_speech_ms: Milliseconds of audio to keep in the pre-speech buffer
            sample_rate: Audio sample rate in Hz
            bytes_per_sample: Number of bytes per audio sample (2 for 16-bit PCM)
            channels: Number of audio channels
            verbose: Whether to log detailed information
        """
        self.pre_speech_ms = pre_speech_ms
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels
        self.verbose = verbose
        
        # Calculate the pre-speech buffer size in bytes
        self.pre_speech_size = int(pre_speech_ms * sample_rate * bytes_per_sample * channels / 1000)
        
        # Initialize buffers
        self.pre_speech_buffer: Deque[bytes] = collections.deque(maxlen=self.pre_speech_size)
        self.speech_buffer: bytes = bytes()
        
        # State tracking
        self.was_speaking = False
        self.is_speaking = False
        self.speech_chunks_count = 0
        
        if verbose:
            console.print(f"[cyan]Speech buffer initialized[/cyan] with {pre_speech_ms}ms pre-speech buffer")
            console.print(f"Pre-speech buffer size: {self.pre_speech_size} bytes")
    
    def add_chunk(self, chunk: bytes, speech_detected: bool) -> Optional[bytes]:
        """
        Add an audio chunk to the buffer and process speech detection.
        
        Args:
            chunk: Audio data in bytes (16-bit PCM)
            speech_detected: Whether VAD detected speech in this chunk
            
        Returns:
            bytes or None: Complete speech segment when speech ends, otherwise None
        """
        # Update state
        self.was_speaking = self.is_speaking
        self.is_speaking = speech_detected
        
        if speech_detected:
            # If this is the start of speech, grab pre-speech buffer
            if not self.was_speaking:
                if self.verbose:
                    console.print("[green]🎙️ Speech started, capturing pre-speech buffer[/green]")
                
                # Initialize the speech buffer with pre-speech data
                self.speech_buffer = bytes()
                for pre_chunk in self.pre_speech_buffer:
                    self.speech_buffer += pre_chunk
                
                self.speech_chunks_count = 1
            
            # Add current chunk to speech buffer
            self.speech_buffer += chunk
            self.speech_chunks_count += 1
            
            # Also keep updating pre-speech buffer even during speech
            # This ensures we have context if a new utterance starts right after this one
            if len(chunk) <= self.pre_speech_size:
                self.pre_speech_buffer.append(chunk)
            else:
                # If chunk is larger than buffer size, add it in pieces
                for i in range(0, len(chunk), self.pre_speech_size):
                    self.pre_speech_buffer.append(chunk[i:i+self.pre_speech_size])
            
            return None
        else:  # No speech detected
            # If speech just ended, return the complete segment
            if self.was_speaking:
                if self.verbose:
                    console.print(f"[blue]🔇 Speech ended, returning segment ({len(self.speech_buffer)} bytes, {self.speech_chunks_count} chunks)[/blue]")
                
                # Get the completed speech segment
                completed_segment = self.speech_buffer
                
                # Reset the speech buffer but keep the pre-speech buffer
                self.speech_buffer = bytes()
                self.speech_chunks_count = 0
                
                return completed_segment
            else:
                # No speech, just update the pre-speech buffer
                if len(chunk) <= self.pre_speech_size:
                    self.pre_speech_buffer.append(chunk)
                else:
                    # If chunk is larger than buffer size, add it in pieces
                    for i in range(0, len(chunk), self.pre_speech_size):
                        self.pre_speech_buffer.append(chunk[i:i+self.pre_speech_size])
                
                return None
    
    def get_current_speech(self) -> bytes:
        """
        Get the current speech buffer without resetting it.
        Useful for getting a speech segment that hasn't ended yet.
        
        Returns:
            bytes: Current speech buffer content
        """
        return self.speech_buffer
    
    def reset(self):
        """
        Reset the speech buffer, clearing all collected audio.
        """
        self.pre_speech_buffer.clear()
        self.speech_buffer = bytes()
        self.was_speaking = False
        self.is_speaking = False
        self.speech_chunks_count = 0
        
        if self.verbose:
            console.print("[yellow]Speech buffer reset[/yellow]") 