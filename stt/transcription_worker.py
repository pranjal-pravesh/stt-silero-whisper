#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transcription worker module for converting speech to text.
Uses faster-whisper for speech recognition.
"""

import os
import tempfile
import wave
import struct
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from rich.console import Console

# Create a console for Rich output
console = Console()

# Import config manager if available
try:
    from config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    console.print("[bold yellow]Config module not found. Using default settings.[/bold yellow]")

# Try to import faster-whisper, but provide fallback if not available
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    console.print("[bold yellow]faster-whisper not found. Using placeholder implementation.[/bold yellow]")


class TranscriptionWorker:
    """
    Worker class to handle speech-to-text transcription.
    Uses faster-whisper for efficient speech recognition.
    """
    
    def __init__(self, model_size: str = "base", language: str = "en", 
                 use_gpu: bool = False, verbose: bool = False):
        """
        Initialize the transcription worker with the specified model.
        
        Args:
            model_size: Whisper model size, one of: "tiny", "base", "small", "medium", "large-v3"
                       Note: Do not use "whisper-" prefix, just the model size name
            language: Language code for transcription (e.g., "en" for English)
            use_gpu: Whether to use GPU for transcription
            verbose: Whether to log detailed information
        """
        # List of valid model sizes for faster-whisper
        VALID_MODELS = [
            "tiny", "tiny.en", 
            "base", "base.en", 
            "small", "small.en", 
            "medium", "medium.en", 
            "large-v1", "large-v2", "large-v3", "large",
            "distil-large-v2", "distil-medium.en", "distil-small.en", 
            "distil-large-v3", "large-v3-turbo", "turbo"
        ]
        
        # Load config if available
        if CONFIG_AVAILABLE:
            config = get_config()
            self.model_name = model_size or config.get("stt.model", "base") 
            
            # Remove any "whisper-" prefix if present (common mistake)
            if self.model_name.startswith("whisper-"):
                original_name = self.model_name
                self.model_name = self.model_name.replace("whisper-", "")
                if verbose:
                    console.print(f"[yellow]Warning: Changed model name from '{original_name}' to '{self.model_name}'[/yellow]")
            
            # Validate model name
            if self.model_name not in VALID_MODELS:
                console.print(f"[bold yellow]Warning: Model '{self.model_name}' may not be valid. Valid models: {', '.join(VALID_MODELS)}[/bold yellow]")
                
            self.language = language or config.get("stt.language", "en")
            self.use_gpu = use_gpu if use_gpu is not None else config.get("vad.use_gpu", False)
        else:
            self.model_name = model_size
            # Remove any "whisper-" prefix if present
            if self.model_name.startswith("whisper-"):
                self.model_name = self.model_name.replace("whisper-", "")
            self.language = language
            self.use_gpu = use_gpu
            
        self.verbose = verbose
        self.model = None
        
        # Initialize the model if faster-whisper is available
        if FASTER_WHISPER_AVAILABLE:
            try:
                if self.verbose:
                    console.print(f"[cyan]Loading Whisper model: {self.model_name}[/cyan]")
                
                # Setup compute type based on GPU availability
                compute_type = "float16" if self.use_gpu else "int8"
                device = "cuda" if self.use_gpu and self.check_gpu_available() else "cpu"
                
                # Initialize the Whisper model
                self.model = WhisperModel(
                    model_size_or_path=self.model_name,
                    device=device,
                    compute_type=compute_type
                )
                
                if self.verbose:
                    console.print(f"[bold green]Whisper model loaded[/bold green] ({self.model_name}, device={device}, compute_type={compute_type})")
            except Exception as e:
                console.print(f"[bold red]Failed to initialize Whisper model: {str(e)}[/bold red]")
                self.model = None
        else:
            console.print("[yellow]Running in placeholder mode - no actual transcription will occur[/yellow]")
    
    def check_gpu_available(self) -> bool:
        """
        Check if GPU is available for computation.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        if FASTER_WHISPER_AVAILABLE:
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return False
    
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Audio data in 16-bit PCM format
            sample_rate: Audio sample rate in Hz
            
        Returns:
            str: Transcribed text
        """
        if not audio_bytes:
            return ""
            
        start_time = time.time()
        audio_duration = len(audio_bytes) / 2 / sample_rate  # in seconds
        
        if self.verbose:
            console.print(f"[cyan]Transcribing {len(audio_bytes)} bytes ({audio_duration:.2f}s) of audio...[/cyan]")
        
        # Check if we have any audio to transcribe
        if not audio_bytes or len(audio_bytes) < sample_rate:  # Less than 1 second
            if self.verbose:
                console.print("[yellow]Audio too short, skipping transcription[/yellow]")
            return ""
            
        try:
            # Write audio bytes to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                self._write_wav(temp_file, audio_bytes, sample_rate)
            
            # Transcribe the audio
            if FASTER_WHISPER_AVAILABLE and self.model:
                segments, info = self.model.transcribe(
                    temp_path, 
                    language=self.language,
                    task="transcribe",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Collect the transcript
                transcript = " ".join(segment.text for segment in segments)
                
                # Log performance stats
                if self.verbose:
                    elapsed = time.time() - start_time
                    real_time_factor = elapsed / audio_duration if audio_duration > 0 else 0
                    console.print(f"[bold green]Transcription complete[/bold green] in {elapsed:.2f}s (RTF: {real_time_factor:.2f})")
                    if hasattr(info, 'language') and info.language:
                        console.print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
                    console.print(f"[bold]Transcript:[/bold] {transcript}")
                
                return transcript
            else:
                # Placeholder implementation if faster-whisper is not available
                if self.verbose:
                    console.print("[yellow]Using placeholder implementation - returning dummy text[/yellow]")
                placeholder = f"This is a placeholder transcription for {audio_duration:.1f} seconds of audio."
                return placeholder
                
        except Exception as e:
            console.print(f"[bold red]Error in transcription: {str(e)}[/bold red]")
            return f"[Transcription error: {str(e)}]"
        finally:
            # Clean up the temporary file
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
    
    def _write_wav(self, file, audio_bytes: bytes, sample_rate: int = 16000):
        """
        Write PCM audio bytes to a WAV file.
        
        Args:
            file: File object to write to
            audio_bytes: Audio data in 16-bit PCM format
            sample_rate: Audio sample rate in Hz
        """
        with wave.open(file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
            
        if self.verbose:
            console.print(f"[blue]Wrote {len(audio_bytes)} bytes to temporary WAV file[/blue]")
    
    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert 16-bit PCM bytes to float32 numpy array.
        
        Args:
            audio_bytes: Audio data in 16-bit PCM format
            
        Returns:
            numpy.ndarray: Audio data as float32 array in range [-1, 1]
        """
        # Calculate number of samples (each sample is 2 bytes for 16-bit audio)
        num_samples = len(audio_bytes) // 2
        
        # Unpack bytes to 16-bit integers
        format_str = f"{num_samples}h"
        try:
            int_samples = struct.unpack(format_str, audio_bytes)
            
            # Convert to float32 array and normalize to [-1, 1]
            float_samples = np.array(int_samples, dtype=np.float32) / 32768.0
            return float_samples
        except Exception as e:
            console.print(f"[bold red]Error converting audio bytes to numpy: {str(e)}[/bold red]")
            return np.array([], dtype=np.float32) 