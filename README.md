# Simple Real-TIme Speech to Text

A Python-based audio processing application that can capture and process microphone input.

## Features

- Capture audio from any connected microphone
- Select specific microphone by index
- 16-bit mono PCM audio at 16000 Hz
- Real-time audio processing

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python main.py
   ```

## Project Structure

```
root/
├── main.py                    # Entry point: ties everything together
├── audio/
│   └── audio_stream.py        # Handles mic input
├── requirements.txt           # For dependencies
└── README.md                  # Project overview
```

## Usage

When you run `main.py`, the application will:

1. List all available microphones with their indices
2. Allow you to select a specific microphone by entering its index
3. Start capturing audio from the selected microphone
4. Process the audio in real-time (future implementation)

To stop the application, press `Ctrl+C`. 
