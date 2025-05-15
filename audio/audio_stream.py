#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
import sys
import time
import numpy as np
import struct
from rich.console import Console

# Create a console for Rich output
console = Console()

# Import VADDetector only if available, otherwise gracefully handle import
try:
    from vad.vad_detector import VADDetector
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    console.print("[bold yellow]VAD module not found. Speech detection will be disabled.[/bold yellow]")

# Import config manager if available
try:
    from config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    console.print("[bold yellow]Config module not found. Using default settings.[/bold yellow]")

class AudioStream:
    """
    Continuously reads audio from the system microphone in small chunks.
    Provides access to audio in 16-bit mono PCM format at 16000 Hz.
    Can optionally detect speech using Silero VAD in real-time.
    """
    
    def __init__(self, chunk_duration_ms=None, sample_rate=None, channels=None, mic_index=None, vad_threshold=None, enable_vad=True, verbose=False):
        """
        Initialize the audio stream with the specified parameters.
        If parameters are None, they will be loaded from config.
        
        Args:
            chunk_duration_ms (int, optional): Duration of each audio chunk in milliseconds.
                                            100ms is recommended for Silero VAD.
            sample_rate (int, optional): Sample rate in Hz
            channels (int, optional): Number of audio channels (1 for mono)
            mic_index (int, optional): Index of the microphone to use. If None, default mic is used.
            vad_threshold (float, optional): Speech detection threshold between 0 and 1. Default is 0.5.
            enable_vad (bool): Whether to enable speech detection using Silero VAD. Default is True.
            verbose (bool): Whether to log detailed information. Default is False.
        """
        # Load config if available
        if CONFIG_AVAILABLE:
            config = get_config()
            # Use passed parameters or fallback to config values
            self.sample_rate = sample_rate or config.get("audio.sample_rate", 16000)
            self.channels = channels or config.get("audio.channels", 1)
            self.mic_index = mic_index if mic_index is not None else config.get("audio.device_index")
            self.vad_threshold = vad_threshold if vad_threshold is not None else config.get("vad.silero.threshold", 0.5)
            self.normalize_audio = config.get("audio.normalize_audio", True)
            
            # Set sample width from config (in bytes)
            sample_width_bytes = config.get("audio.sample_width", 2)
            if sample_width_bytes == 2:
                self.format = pyaudio.paInt16
            elif sample_width_bytes == 4:
                self.format = pyaudio.paInt32
            elif sample_width_bytes == 1:
                self.format = pyaudio.paInt8
            else:
                self.format = pyaudio.paInt16  # Default to 16-bit PCM
                console.print(f"[yellow]Unsupported sample width: {sample_width_bytes}, defaulting to 16-bit[/yellow]")
            
            # Calculate frames per buffer, respecting config or chunk_duration_ms if provided
            if chunk_duration_ms is not None:
                self.frames_per_buffer = int(self.sample_rate * chunk_duration_ms / 1000)
            else:
                self.frames_per_buffer = config.get("audio.frames_per_buffer", 
                                                   config.get("audio.chunk_size", 480))
        else:
            # Use passed parameters or fallback to hardcoded defaults
            self.sample_rate = sample_rate or 16000
            self.channels = channels or 1
            self.mic_index = mic_index
            self.vad_threshold = vad_threshold or 0.25
            self.format = pyaudio.paInt16  # 16-bit PCM
            self.normalize_audio = True
            
            # Calculate frames per buffer
            if chunk_duration_ms is None:
                self.frames_per_buffer = 480  # Default to 480 samples (30ms at 16kHz)
            else:
                self.frames_per_buffer = int(self.sample_rate * chunk_duration_ms / 1000)
        
        self.enable_vad = enable_vad and VAD_AVAILABLE
        self.verbose = verbose
        
        self.pa = None
        self.stream = None
        self.vad = None
        self.buffer = bytes()
        
        try:
            self.pa = pyaudio.PyAudio()
            
            # List available microphones if requested
            self._list_microphones()
            
            # Open the audio stream with the specified microphone
            self.stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.mic_index,
                frames_per_buffer=self.frames_per_buffer
            )
            
            # Initialize VAD if enabled
            if self.enable_vad:
                try:
                    self.vad = VADDetector(threshold=self.vad_threshold, sample_rate=self.sample_rate, verbose=self.verbose)
                    console.print(f"[bold green]Voice Activity Detection enabled[/bold green] with threshold {self.vad_threshold}")
                    # Get the required buffer size based on VAD requirements
                    self.required_samples = self.vad.required_samples
                    self.min_buffer_size = self.required_samples * 2  # in bytes (each sample is 2 bytes)
                except Exception as e:
                    console.print(f"[bold red]Failed to initialize VAD: {str(e)}[/bold red]")
                    self.enable_vad = False
            
            device_info = "default microphone"
            if self.mic_index is not None:
                try:
                    device_info = self.pa.get_device_info_by_index(self.mic_index)['name']
                except:
                    device_info = f"microphone at index {self.mic_index}"
                    
            console.print(f"[bold green]Audio stream started:[/bold green] {self.channels} channel(s) at {self.sample_rate} Hz using {device_info}")
            console.print(f"Chunk duration: {1000 * self.frames_per_buffer / self.sample_rate:.1f}ms ({self.frames_per_buffer} samples)")
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to initialize audio stream: {str(e)}")
    
    def _list_microphones(self):
        """List all available microphones."""
        console.print("\n[bold cyan]Available microphones:[/bold cyan]")
        console.rule()
        
        for i in range(self.pa.get_device_count()):
            try:
                device_info = self.pa.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:  # Only show input devices
                    console.print(f"[bold][{i}][/bold] {device_info['name']}")
                    console.print(f"    Channels: {device_info['maxInputChannels']}")
                    console.print(f"    Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
                    console.rule()
            except Exception as e:
                console.print(f"[bold red][{i}] Error getting device info: {str(e)}[/bold red]")
                console.rule()
        
        console.print("")
    
    @classmethod
    def list_available_microphones(cls):
        """
        Static method to list all available microphones without creating an AudioStream.
        
        Returns:
            dict: Dictionary of available microphones with index as key
        """
        mics = {}
        p = pyaudio.PyAudio()
        
        console.print("\n[bold cyan]Available microphones:[/bold cyan]")
        console.rule()
        
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:  # Only show input devices
                    mics[i] = device_info
                    console.print(f"[bold][{i}][/bold] {device_info['name']}")
                    console.print(f"    Channels: {device_info['maxInputChannels']}")
                    console.print(f"    Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
                    console.rule()
            except Exception as e:
                console.print(f"[bold red][{i}] Error getting device info: {str(e)}[/bold red]")
                console.rule()
        
        p.terminate()
        console.print("")
        return mics
    
    def _bytes_to_numpy(self, audio_chunk):
        """
        Convert 16-bit PCM bytes to float32 numpy array in range [-1.0, 1.0].
        
        Args:
            audio_chunk (bytes): 16-bit mono PCM audio data
            
        Returns:
            numpy.ndarray: Float32 audio data in range [-1.0, 1.0]
        """
        try:
            # Each sample is 2 bytes (16-bit)
            num_samples = len(audio_chunk) // 2
            
            # Unpack the bytes to 16-bit integers
            format_str = f"{num_samples}h"
            int_samples = struct.unpack(format_str, audio_chunk)
            
            # Convert to float32 numpy array and normalize to [-1.0, 1.0]
            float_samples = np.array(int_samples, dtype=np.float32) / 32768.0
            
            return float_samples
        except Exception as e:
            console.print(f"[bold red]Error converting audio to numpy: {str(e)}[/bold red]")
            return np.array([], dtype=np.float32)
    
    def get_next_chunk(self):
        """
        Generator that yields the next audio buffer and performs speech detection if enabled.
        
        Yields:
            tuple: (bytes, bool or None) Audio data in PCM format and speech detection result
                  (True if speech detected, False if no speech, None if VAD is disabled)
        """
        if not self.stream:
            raise RuntimeError("Audio stream is not initialized")
        
        try:
            chunk_count = 0
            if self.verbose:
                console.print("[bold green]Starting audio capture...[/bold green]")
            else:
                # Still log this important message but simpler
                console.print("Starting audio capture...")
            
            while self.stream.is_active():
                # Read audio data from the stream
                data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
                
                # Only print log message occasionally to reduce output
                chunk_count += 1
                if self.verbose and chunk_count % 100 == 1:
                    console.print(f"Processing audio... ({len(data)} bytes per chunk)")
                
                # Speech detection
                speech_detected = None
                
                if self.enable_vad and self.vad is not None:
                    try:
                        # Add new audio to buffer
                        self.buffer += data
                        
                        # Process buffer if it's large enough
                        if len(self.buffer) >= self.min_buffer_size:
                            # Check if the chunk contains speech
                            speech_detected = self.vad.is_speech(self.buffer)
                            
                            # Keep a sliding window buffer (retain the latter portion)
                            if len(self.buffer) > self.min_buffer_size * 2:
                                self.buffer = self.buffer[-self.min_buffer_size:]
                    except Exception as e:
                        console.print(f"[bold red]Error in speech detection: {str(e)}[/bold red]")
                        # If we get persistent errors, disable VAD
                        self.vad_error_count = getattr(self, 'vad_error_count', 0) + 1
                        if self.vad_error_count > 10:
                            console.print("[bold red]Too many VAD errors, disabling speech detection[/bold red]")
                            self.enable_vad = False
                            self.vad = None
                        # Continue without speech detection for this chunk
                        speech_detected = None
                
                yield data, speech_detected
                
        except Exception as e:
            console.print(f"[bold red]Error reading from audio stream: {str(e)}[/bold red]")
            raise
    
    def _cleanup(self):
        """Clean up resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                console.print("[bold green]Audio stream stopped[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error stopping audio stream: {str(e)}[/bold red]")
            finally:
                self.stream = None
        
        if self.pa:
            try:
                self.pa.terminate()
            except Exception as e:
                console.print(f"[bold red]Error terminating PyAudio: {str(e)}[/bold red]")
            finally:
                self.pa = None
        
        # Reset VAD if initialized
        if self.vad is not None:
            try:
                self.vad.reset()
            except Exception as e:
                console.print(f"[bold red]Error resetting VAD: {str(e)}[/bold red]")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
    
    def __del__(self):
        """Destructor to ensure resources are properly cleaned up."""
        self._cleanup()


if __name__ == "__main__":
    console.print("Testing AudioStream with Silero VAD - Press Ctrl+C to stop")
    
    # List available microphones
    mics = AudioStream.list_available_microphones()
    
    # Ask user to select a microphone
    selected_mic = None
    if mics:
        try:
            mic_index = input("Enter microphone index to use (leave empty for default): ").strip()
            if mic_index:
                selected_mic = int(mic_index)
        except ValueError:
            console.print("[bold yellow]Invalid input, using default microphone[/bold yellow]")
    
    try:
        # Use the selected microphone with VAD enabled
        with AudioStream(chunk_duration_ms=100, mic_index=selected_mic, enable_vad=True) as audio:
            console.print("Listening for audio and detecting speech... (Press Ctrl+C to stop)")
            
            # Process audio for 10 seconds
            start_time = time.time()
            for chunk, is_speech in audio.get_next_chunk():
                # Print detection result if available
                if is_speech is not None:
                    if is_speech:
                        console.print("[bold green]Speech detected in this chunk![/bold green]")
                    else:
                        console.print("[bold yellow]No speech in this chunk.[/bold yellow]")
                
                # Stop after 10 seconds
                if time.time() - start_time > 10:
                    break
    except KeyboardInterrupt:
        console.print("\nStopped by user")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]") 