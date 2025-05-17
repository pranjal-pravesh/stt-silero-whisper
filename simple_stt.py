import torch
import numpy as np
import sounddevice as sd
import queue
import time
from faster_whisper import WhisperModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# --- Constants ---
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

console = Console()

# --- List audio input devices and select one ---
console.print("[bold cyan]Available audio input devices:[/bold cyan]")
input_devices = []
for idx, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        input_devices.append(idx)
        console.print(f"[green]{idx}[/green]: {device['name']}")

device_index = None
while device_index not in input_devices:
    try:
        device_index = int(console.input("[bold yellow]Select microphone device index:[/bold yellow] "))
        if device_index not in input_devices:
            console.print("[red]Invalid device index, please try again.[/red]")
    except ValueError:
        console.print("[red]Please enter a valid integer.[/red]")

# --- Load Silero VAD ---
console.print("[bold cyan]Loading Silero VAD model...[/bold cyan]")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False,
    verbose=False
)
get_speech_timestamps, _, read_audio, _, _ = utils

# --- Load Whisper ---
console.print("[bold cyan]Loading Whisper model (faster-whisper)...[/bold cyan]")
whisper_model = WhisperModel("small.en", compute_type="int8", device="cpu")

# --- Audio Queue ---
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        console.log(f"[yellow]Audio stream status: {status}[/yellow]")
    audio_q.put(indata.copy())

# --- Start Input Stream ---
console.print(f"[bold green]Starting audio stream on device {device_index}...[/bold green]")
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCK_SIZE,
    device=device_index,
    callback=audio_callback
)
stream.start()

buffered_audio = []
speech_active = False
last_speech_time = time.time()
last_transcript = ""

def render_panel():
    status_text = "Listening for speech..." if not speech_active else "[bold green]Speech detected! Buffering...[/bold green]"
    if speech_active and time.time() - last_speech_time > 1.0:
        status_text = "[bold yellow]Speech ended. Transcribing...[/bold yellow]"
    text = Text()
    text.append(f"Status: {status_text}\n", style="bold magenta")

    if buffered_audio:
        text.append(f"Buffered audio length: {sum(len(chunk) for chunk in buffered_audio)} samples\n", style="cyan")
    else:
        text.append("Buffered audio length: 0 samples\n", style="cyan")

    text.append(f"Last transcript:\n", style="bold white")
    if last_transcript:
        text.append(f"{last_transcript}\n", style="bright_white")
    else:
        text.append("None yet\n", style="dim")

    return Panel(text, title="[bold blue]Real-time Speech Transcription[/bold blue]", border_style="blue")

with Live(render_panel(), refresh_per_second=4, console=console) as live:
    try:
        while True:
            audio_chunk = audio_q.get()
            audio_np = audio_chunk[:, 0]
            temp_wav = torch.from_numpy(audio_np).float()

            speech_timestamps = get_speech_timestamps(
                temp_wav, model, sampling_rate=SAMPLE_RATE, threshold=0.2
            )

            # Update speech state and buffer
            if speech_timestamps:
                buffered_audio.append(audio_np)
                speech_active = True
                last_speech_time = time.time()
            elif speech_active and time.time() - last_speech_time > 1.0:
                # Speech ended, transcribe
                live.update(render_panel())
                console.print("[bold yellow]Speech ended. Transcribing...[/bold yellow]")
                full_audio = np.concatenate(buffered_audio)
                segments, _ = whisper_model.transcribe(full_audio, language="en")
                combined_text = " ".join(segment.text for segment in segments)
                last_transcript = combined_text.strip()
                console.print(f"[bold green]Transcription:[/bold green] {last_transcript}")

                buffered_audio = []
                speech_active = False

            live.update(render_panel())

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping...[/bold red]")
        stream.stop()
        stream.close()
