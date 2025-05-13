#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the Proactive Agent application.
This module ties together all components of the system.
"""

import sys
import os
import time
import threading
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.table import Table
from rich.syntax import Syntax

# Remove the parent directory addition since we're now in the root directory
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio import AudioStream
from config import get_config
from stt import SpeechBuffer, TranscriptionWorker


def main():
    """Main function to run the proactive agent."""
    # Initialize Rich console
    console = Console()
    
    try:
        console.print("[bold green]Starting Proactive Agent...[/bold green]")
        
        # Load configuration
        config = get_config()
        
        # List available microphones and let the user select one
        mics = AudioStream.list_available_microphones()
        
        # Ask user to select a microphone
        selected_mic = config.get("audio.device_index")  # Get from config
        if mics:
            try:
                mic_index = console.input("\n[bold cyan]Enter microphone index to use (leave empty for default): [/bold cyan]").strip()
                if mic_index:
                    selected_mic = int(mic_index)
            except ValueError:
                console.print("[bold yellow]Invalid input, using default microphone[/bold yellow]")
        
        # Set verbosity for cleaner output
        verbose = False  # Change back to False to reduce terminal noise
        
        # Initialize the audio stream with the selected microphone and VAD enabled
        with AudioStream(
            chunk_duration_ms=int(1000 * config.get("audio.chunk_size") / config.get("audio.sample_rate", 16000)),
            sample_rate=config.get("audio.sample_rate", 16000),
            channels=config.get("audio.channels", 1),
            mic_index=selected_mic,
            vad_threshold=config.get("vad.silero.threshold", 0.5),
            enable_vad=True,
            verbose=verbose
        ) as audio:
            console.print("\n[bold green]Listening for audio input. Press Ctrl+C to stop.[/bold green]")
            
            # Initialize speech buffer for collecting speech segments
            speech_buffer = SpeechBuffer(
                pre_speech_ms=500,
                sample_rate=config.get("audio.sample_rate", 16000),
                verbose=verbose
            )
            
            # Initialize transcription worker
            transcriber = TranscriptionWorker(
                model_size=config.get("stt.model", "base"),
                language=config.get("stt.language", "en"),
                use_gpu=config.get("vad.use_gpu", False),
                verbose=verbose
            )
            
            # Status tracking variables
            speech_count = 0
            is_speaking = False
            start_time = time.time()
            last_vad_prob = 0.0
            speech_duration = 0
            silence_duration = 0
            last_status_change = time.time()
            current_transcript = ""
            transcription_queue = []
            
            # Create a spinner
            spinner = Spinner("dots", text="Listening...")
            
            # Transcription worker thread
            def transcription_worker():
                while True:
                    # Check if there's a segment to transcribe
                    if transcription_queue:
                        audio_segment, segment_id = transcription_queue.pop(0)
                        try:
                            # Transcribe the audio segment
                            transcript = transcriber.transcribe(
                                audio_segment, 
                                sample_rate=config.get("audio.sample_rate", 16000)
                            )
                            
                            # Update the current transcript
                            nonlocal current_transcript
                            current_transcript = transcript
                        except Exception as e:
                            console.print(f"[bold red]Transcription error: {str(e)}[/bold red]")
                    
                    # Sleep to avoid busy waiting
                    time.sleep(0.1)
            
            # Start the transcription worker thread
            transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
            transcription_thread.start()
            
            # Create a single live display that will be continuously updated
            with Live(
                Panel(Text("Initializing..."), title="Proactive Agent", border_style="green"),
                refresh_per_second=4,  # Reduce to 4 to be less resource intensive
                console=console,
                auto_refresh=True,
                transient=False  # Important: This ensures we keep a single display
            ) as live:
                
                def update_display():
                    """Update the live display with current status."""
                    current_time = time.time()
                    elapsed = int(current_time - start_time)
                    minutes, seconds = divmod(elapsed, 60)
                    
                    # Create a table for the display
                    table = Table(show_header=False, box=None, padding=(0, 1))
                    table.add_column("Label", style="cyan")
                    table.add_column("Value", style="green")
                    
                    # Calculate duration of current state
                    current_state_duration = int(current_time - last_status_change)
                    current_mins, current_secs = divmod(current_state_duration, 60)
                    
                    if is_speaking:
                        status_icon = "🎤"
                        status_text = f"[bold yellow]Speech detected![/bold yellow]"
                        table.add_row("Current state:", f"{status_icon} {status_text} ({current_mins:02d}:{current_secs:02d})")
                        table.add_row("Speech probability:", f"[yellow]{last_vad_prob:.2f}[/yellow]")
                    else:
                        # Use the spinner properly
                        frame = spinner.frames[int(current_time * 10) % len(spinner.frames)]
                        status_text = f"[blue]Listening for speech...[/blue]"
                        table.add_row("Current state:", f"{frame} {status_text} ({current_mins:02d}:{current_secs:02d})")
                        if last_vad_prob > 0:
                            table.add_row("Speech probability:", f"[dim]{last_vad_prob:.2f}[/dim]")
                    
                    # Add stats to the table
                    table.add_row("Detected segments:", f"{speech_count}")
                    
                    # Format durations - Make sure to convert floats to integers before formatting
                    if speech_duration > 0:
                        speech_mins, speech_secs = divmod(int(speech_duration), 60)
                        table.add_row("Total speech time:", f"{int(speech_mins):02d}:{int(speech_secs):02d}")
                    
                    if silence_duration > 0:
                        silence_mins, silence_secs = divmod(int(silence_duration), 60)
                        table.add_row("Total silence time:", f"{int(silence_mins):02d}:{int(silence_secs):02d}")
                    
                    table.add_row("Session time:", f"{minutes:02d}:{seconds:02d}")
                    
                    # Add the transcript if available
                    if current_transcript:
                        table.add_row("", "")  # Empty row for spacing
                        table.add_row("[bold]Last transcript:[/bold]", "")
                        transcript_style = Syntax(
                            current_transcript, 
                            "text", 
                            theme="monokai",
                            word_wrap=True,
                            line_numbers=False,
                            background_color="default"
                        )
                        table.add_row("", transcript_style)
                    
                    return Panel(
                        table,
                        title="Proactive Agent",
                        border_style="green"
                    )
                
                # Initial update
                live.update(update_display())
                
                # Process audio chunks with built-in VAD
                for chunk, is_speech in audio.get_next_chunk():
                    current_time = time.time()
                    
                    # Get speech probability from VAD if available
                    vad_prob = getattr(audio.vad, 'last_speech_probability', 0.0) if audio.vad else 0.0
                    if vad_prob is not None:
                        last_vad_prob = vad_prob
                    
                    # Add chunk to speech buffer and check if a segment is complete
                    audio_segment = speech_buffer.add_chunk(chunk, is_speech)
                    if audio_segment:
                        # Add the segment to the transcription queue
                        transcription_queue.append((audio_segment, speech_count))
                    
                    # Update the speaking status based on VAD result
                    if is_speech:
                        if not is_speaking:  # Only increment counter when we transition to speaking
                            speech_count += 1
                            last_status_change = current_time
                        is_speaking = True
                        speech_duration += 0.1  # More accurate time increment based on 100ms chunks
                    else:
                        if is_speaking:  # Track when we transition to silence
                            last_status_change = current_time
                        is_speaking = False
                        silence_duration += 0.1  # More accurate time increment based on 100ms chunks
                    
                    # Update the display
                    live.update(update_display())
                
    except KeyboardInterrupt:
        console.print("\n[bold red]Proactive Agent stopped by user[/bold red]", highlight=False)
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]", highlight=False)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 