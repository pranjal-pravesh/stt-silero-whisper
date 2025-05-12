#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the Proactive Agent application.
This module ties together all components of the system.
"""

import sys
import os

# Remove the parent directory addition since we're now in the root directory
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.audio_stream import AudioStream


def main():
    """Main function to run the proactive agent."""
    try:
        print("Starting Proactive Agent...")
        
        # List available microphones and let the user select one
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
        
        # Initialize the audio stream with the selected microphone
        with AudioStream(mic_index=selected_mic) as audio:
            print("Listening for audio input. Press Ctrl+C to stop.")
            for chunk in audio.get_next_chunk():
                # Process the audio chunk (to be implemented)
                pass
                
    except KeyboardInterrupt:
        print("\nProactive Agent stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 