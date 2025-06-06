#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice Activity Detection (VAD) module using Silero VAD.
This module provides functionality to detect speech in audio chunks.
"""

import sys
import torch
import numpy as np
import struct
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

class VADDetector:
    """
    Voice Activity Detection using the Silero VAD model.
    Detects whether speech is present in an audio chunk.
    """
    
    def __init__(self, threshold=None, sample_rate=None, verbose=False):
        """
        Initialize the VAD detector with the specified threshold.
        If parameters are None, they will be loaded from config.
        
        Args:
            threshold (float, optional): Speech detection threshold between 0 and 1.
                                      Higher values make detection more strict. Default is 0.5.
            sample_rate (int, optional): Sample rate in Hz. Default is 16000.
            verbose (bool): Whether to log detailed information. Default is False.
        """
        # Load configuration from config manager
        if CONFIG_AVAILABLE:
            config = get_config()
            vad_type = config.get("vad.type", "silero")
            self.threshold = threshold if threshold is not None else config.get(f"vad.{vad_type}.threshold", 0.5)
            self.sample_rate = sample_rate or config.get("audio.sample_rate", 16000)
            self.window_size_samples = config.get(f"vad.{vad_type}.window_size_samples", 512)
            self.min_speech_duration_ms = config.get(f"vad.{vad_type}.min_speech_duration_ms", 250)
            self.min_silence_duration_ms = config.get(f"vad.{vad_type}.min_silence_duration_ms", 100)
            self.speech_pad_ms = config.get(f"vad.{vad_type}.speech_pad_ms", 30)
            self.use_gpu = config.get("vad.use_gpu", False)
        else:
            # Fallback to default values
            self.threshold = threshold if threshold is not None else 0.5
            self.sample_rate = sample_rate or 16000
            self.window_size_samples = 512
            self.min_speech_duration_ms = 250
            self.min_silence_duration_ms = 100
            self.speech_pad_ms = 30
            self.use_gpu = False
        
        self.verbose = verbose
        self.last_speech_probability = 0.0  # Track last speech probability
        
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        # Silero VAD requires specific number of samples:
        # - 256 samples for 8kHz
        # - 512 samples for 16kHz
        if self.window_size_samples not in [256, 512, 1024, 1536]:
            console.print(f"[yellow]Warning: window_size_samples should be 256, 512, 1024, or 1536. Using default instead of {self.window_size_samples}[/yellow]")
            self.required_samples = 512 if self.sample_rate == 16000 else 256
        else:
            self.required_samples = self.window_size_samples
        
        try:
            # Initialize silero VAD model
            console.print("[bold cyan]Loading Silero VAD model...[/bold cyan]")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,  # Use PyTorch version
                verbose=False  # Reduce logging noise
            )
            
            # Move to GPU if requested and available
            if self.use_gpu and torch.cuda.is_available():
                model = model.to('cuda')
                console.print("[bold green]Using GPU for VAD processing[/bold green]")
            
            # Unpack the utils
            (get_speech_timestamps, 
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks) = utils
            
            # Create VAD iterator with explicit options
            self.model = model  # Store the model directly
            self.vad = VADIterator(
                model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate
            )
            
            # Store the get_speech_timestamps function for possible future use with speech_pad_ms
            self.get_speech_timestamps = get_speech_timestamps
            
            console.print(
                f"[bold green]Silero VAD initialized[/bold green] "
                f"threshold={self.threshold}, sample_rate={self.sample_rate} Hz, "
                f"window_size_samples={self.required_samples}, use_gpu={self.use_gpu}, "
                f"speech_pad_ms={self.speech_pad_ms}"
            )
            
            # Test the VAD with a silent sample to ensure it's working
            test_tensor = torch.zeros(self.required_samples)
            if self.use_gpu and torch.cuda.is_available():
                test_tensor = test_tensor.to('cuda')
            _ = self.vad(test_tensor, self.sample_rate)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Silero VAD: {str(e)}")
    
    def _bytes_to_numpy(self, audio_chunk):
        """
        Convert 16-bit PCM bytes to float32 numpy array in range [-1.0, 1.0].
        
        Args:
            audio_chunk (bytes): 16-bit mono PCM audio data
            
        Returns:
            numpy.ndarray: Float32 audio data in range [-1.0, 1.0]
        """
        # Each sample is 2 bytes (16-bit)
        num_samples = len(audio_chunk) // 2
        
        # Unpack the bytes to 16-bit integers
        format_str = f"{num_samples}h"
        int_samples = struct.unpack(format_str, audio_chunk)
        
        # Convert to float32 numpy array and normalize to [-1.0, 1.0]
        float_samples = np.array(int_samples, dtype=np.float32) / 32768.0
        
        return float_samples
    
    def is_speech(self, audio_chunk):
        """
        Determine if the audio chunk contains speech.
        
        Args:
            audio_chunk (bytes): Audio data in 16-bit mono PCM format.
        
        Returns:
            bool: True if speech is detected, False otherwise
        """
        try:
            # Convert audio bytes to numpy array
            audio_np = self._bytes_to_numpy(audio_chunk)
            
            # Check if we have a valid numpy array
            if audio_np.size == 0:
                if self.verbose:
                    console.print("[yellow]Warning: Empty audio chunk, skipping VAD processing[/yellow]")
                self.last_speech_probability = 0.0
                return False
            
            # Resize array to the required number of samples
            # If too long, take only the needed portion
            # If too short, return False (can't process)
            if len(audio_np) < self.required_samples:
                if self.verbose:
                    console.print(f"[yellow]Warning: Audio chunk too short ({len(audio_np)} samples, need {self.required_samples})[/yellow]")
                self.last_speech_probability = 0.0
                return False
            
            # Process in chunks of required_samples
            # For better accuracy, we'll check multiple chunks and return True if any contain speech
            speech_detected = False
            speech_probs = []
            
            # For more accurate detection with speech_pad_ms, we could use get_speech_timestamps
            # on the entire audio chunk and apply padding, but this would be more computationally expensive
            # For now, we'll use the VAD iterator approach and consider implementing a more
            # sophisticated detection method if needed
            
            # Limit the number of chunks to process to avoid excessive processing
            max_chunks = min(len(audio_np) // self.required_samples, 10)
            
            for i in range(0, min(len(audio_np), max_chunks * self.required_samples), self.required_samples):
                if i + self.required_samples > len(audio_np):
                    break
                    
                # Extract exactly the required number of samples
                chunk = audio_np[i:i+self.required_samples]
                
                # Ensure chunk is the exact size required
                if len(chunk) != self.required_samples:
                    if self.verbose:
                        console.print(f"[yellow]Warning: Chunk size mismatch, got {len(chunk)}, expected {self.required_samples}[/yellow]")
                    continue
                
                # Convert to torch tensor
                try:
                    audio_tensor = torch.from_numpy(chunk)
                except Exception as e:
                    if self.verbose:
                        console.print(f"[bold red]Error converting chunk to tensor: {str(e)}[/bold red]")
                    continue
                
                # Get speech probability from the model
                try:
                    vad_result = self.vad(audio_tensor, self.sample_rate)
                    
                    # If the VAD iterator returns None, fall back to direct model evaluation
                    if vad_result is None:
                        # Direct model call as fallback
                        with torch.no_grad():
                            vad_result = self.model(audio_tensor, self.sample_rate).item()
                    
                    # Add proper type checking to handle unexpected return types
                    if vad_result is None:
                        continue
                    
                    if isinstance(vad_result, dict):
                        # Some versions of the model might return dict with probability
                        if 'probability' in vad_result:
                            speech_prob = float(vad_result['probability'])
                        elif 'start' in vad_result:
                            # Start of speech segment marker, not an error
                            if self.verbose:
                                console.print(f"[blue]🎬 Speech segment starts at {vad_result['start']}s[/blue]")
                            continue
                        elif 'end' in vad_result:
                            # End of speech segment marker, not an error
                            if self.verbose:
                                console.print(f"[blue]🏁 Speech segment ends at {vad_result['end']}s[/blue]")
                            continue
                        else:
                            # Only print warnings for truly unexpected dictionary formats
                            if self.verbose:
                                console.print(f"[yellow]Warning: VAD returned dict with unknown format: {vad_result}[/yellow]")
                            continue
                    elif isinstance(vad_result, (int, float)):
                        # Handle direct float/int values
                        speech_prob = float(vad_result)
                    else:
                        # Normal case: tensor with single value
                        speech_prob = vad_result.item()
                except Exception as e:
                    if self.verbose:
                        console.print(f"[bold red]Warning: Couldn't extract probability from VAD result: {type(vad_result)} - {e}[/bold red]")
                    continue
                
                speech_probs.append(speech_prob)
                
                # Check if speech is detected in this chunk
                if speech_prob > self.threshold:
                    speech_detected = True
            
            # Log the result with probability (average of all chunks)
            avg_prob = 0
            max_prob = 0
            
            if speech_probs:
                avg_prob = sum(speech_probs) / len(speech_probs)
                max_prob = max(speech_probs)
                self.last_speech_probability = max_prob  # Store the maximum probability
            else:
                self.last_speech_probability = 0.0
                
            if self.verbose:
                if speech_detected:
                    console.print(f"[bold green]🗣️ Speech detected[/bold green] (avg_prob={avg_prob:.2f}, max_prob={max_prob:.2f})")
                elif max_prob > 0.1:
                    # Only log silence if there was some non-trivial probability
                    console.print(f"[dim]🔇 Silence[/dim] (avg_prob={avg_prob:.2f}, max_prob={max_prob:.2f})")
            
            return speech_detected
            
        except Exception as e:
            console.print(f"[bold red]Error in VAD processing: {str(e)}[/bold red]")
            self.last_speech_probability = 0.0
            return False
    
    def reset(self):
        """
        Reset the VAD internal state.
        Should be called when processing a new audio stream.
        """
        if hasattr(self, 'vad'):
            try:
                self.vad.reset_states()
                if self.verbose:
                    console.print("[blue]VAD state reset[/blue]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to reset VAD state: {e}[/yellow]")
                # Recreate the VAD iterator if reset fails
                if hasattr(self, 'model'):
                    try:
                        self.vad = VADIterator(self.model, threshold=self.threshold, 
                                              sampling_rate=self.sample_rate)
                        if self.verbose:
                            console.print("[blue]VAD iterator recreated[/blue]")
                    except Exception as e2:
                        console.print(f"[bold red]Error recreating VAD iterator: {e2}[/bold red]")


if __name__ == "__main__":
    # Example usage with AudioStream
    import os
    import sys
    import time
    
    # Add the parent directory to sys.path to enable relative imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from audio.audio_stream import AudioStream
    
    print("Testing Silero VAD Detector with AudioStream - Press Ctrl+C to stop")
    
    try:
        # Initialize VAD with default settings
        vad = VADDetector()
        
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
                print("Invalid input, using default microphone")
        
        # Set chunk duration to 100ms for gathering enough audio
        with AudioStream(chunk_duration_ms=100, mic_index=selected_mic) as audio:
            print("Listening for speech... (Press Ctrl+C to stop)")
            
            buffer = bytes()
            # Calculate buffer size in bytes (each sample is 2 bytes)
            min_buffer_size = vad.required_samples * 2
            
            for chunk in audio.get_next_chunk():
                # Add new audio to buffer
                buffer += chunk
                
                # Process buffer if it's large enough
                if len(buffer) >= min_buffer_size:
                    # Check if the chunk contains speech
                    is_speech = vad.is_speech(buffer)
                    
                    # Keep a sliding window buffer (retain the latter half)
                    if len(buffer) > min_buffer_size * 2:
                        buffer = buffer[-min_buffer_size:]
                
    except KeyboardInterrupt:
        print("\nStopped by user")
        if 'vad' in locals():
            vad.reset()
    except Exception as e:
        print(f"Error: {str(e)}") 