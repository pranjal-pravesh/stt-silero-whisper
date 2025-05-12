#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
import sys
import time

class AudioStream:
    """
    Continuously reads audio from the system microphone in small chunks.
    Provides access to audio in 16-bit mono PCM format at 16000 Hz.
    """
    
    def __init__(self, chunk_duration_ms=30, sample_rate=16000, channels=1, mic_index=None):
        """
        Initialize the audio stream with the specified parameters.
        
        Args:
            chunk_duration_ms (int): Duration of each audio chunk in milliseconds
            sample_rate (int): Sample rate in Hz
            channels (int): Number of audio channels (1 for mono)
            mic_index (int, optional): Index of the microphone to use. If None, default mic is used.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = pyaudio.paInt16  # 16-bit PCM
        self.mic_index = mic_index
        
        # Calculate frames per buffer based on chunk duration
        self.frames_per_buffer = int(sample_rate * chunk_duration_ms / 1000)
        
        self.pa = None
        self.stream = None
        
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
            
            device_info = "default microphone"
            if self.mic_index is not None:
                try:
                    device_info = self.pa.get_device_info_by_index(self.mic_index)['name']
                except:
                    device_info = f"microphone at index {self.mic_index}"
                    
            print(f"Audio stream started: {self.channels} channel(s) at {self.sample_rate} Hz using {device_info}")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to initialize audio stream: {str(e)}")
    
    def _list_microphones(self):
        """List all available microphones."""
        print("\nAvailable microphones:")
        print("-" * 60)
        
        for i in range(self.pa.get_device_count()):
            try:
                device_info = self.pa.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:  # Only show input devices
                    print(f"[{i}] {device_info['name']}")
                    print(f"    Channels: {device_info['maxInputChannels']}")
                    print(f"    Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
                    print("-" * 60)
            except Exception as e:
                print(f"[{i}] Error getting device info: {str(e)}")
                print("-" * 60)
        
        print("")
    
    @classmethod
    def list_available_microphones(cls):
        """
        Static method to list all available microphones without creating an AudioStream.
        
        Returns:
            dict: Dictionary of available microphones with index as key
        """
        mics = {}
        p = pyaudio.PyAudio()
        
        print("\nAvailable microphones:")
        print("-" * 60)
        
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:  # Only show input devices
                    mics[i] = device_info
                    print(f"[{i}] {device_info['name']}")
                    print(f"    Channels: {device_info['maxInputChannels']}")
                    print(f"    Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
                    print("-" * 60)
            except Exception as e:
                print(f"[{i}] Error getting device info: {str(e)}")
                print("-" * 60)
        
        p.terminate()
        print("")
        return mics
    
    def get_next_chunk(self):
        """
        Generator that yields the next audio buffer.
        
        Yields:
            bytes: Audio data in the specified format
        """
        if not self.stream:
            raise RuntimeError("Audio stream is not initialized")
        
        try:
            while self.stream.is_active():
                data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
                print(f"Received audio chunk: {len(data)} bytes")
                yield data
        except Exception as e:
            print(f"Error reading from audio stream: {str(e)}", file=sys.stderr)
            raise
    
    def _cleanup(self):
        """Clean up resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                print("Audio stream stopped")
            except Exception as e:
                print(f"Error stopping audio stream: {str(e)}", file=sys.stderr)
            finally:
                self.stream = None
        
        if self.pa:
            try:
                self.pa.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {str(e)}", file=sys.stderr)
            finally:
                self.pa = None
    
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
    print("Testing AudioStream - Press Ctrl+C to stop")
    
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
    
    try:
        # Use the selected microphone
        with AudioStream(mic_index=selected_mic) as audio:
            # Collect audio for 5 seconds as an example
            start_time = time.time()
            for chunk in audio.get_next_chunk():
                if time.time() - start_time > 5:
                    break
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr) 