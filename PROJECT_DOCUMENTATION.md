# Voice Activity Detection (VAD) System Documentation

## Project Overview

This project implements a real-time Voice Activity Detection (VAD) system using Python. The system captures audio from a microphone, processes it in small chunks, and detects when speech is present in the audio stream.

## System Architecture

The project is organized with a clean separation of concerns:

```
root/
├── main.py                    # Entry point that ties everything together
├── audio/
│   └── audio_stream.py        # Handles microphone input capture
├── vad/
│   └── vad_detector.py        # Speech detection using Silero VAD
├── requirements.txt           # Dependencies
└── README.md                  # Project overview
```

## Components

### 1. AudioStream (audio/audio_stream.py)

**Implementation Details:**
- Uses PyAudio to capture microphone input in real-time
- Configured for 16-bit mono PCM audio at 16kHz
- Captures audio in small chunks (default: 100ms)
- Implements context manager pattern for proper resource management
- Provides ability to list and select from available microphones

**Design Decisions:**
- Chunk duration of 100ms chosen as optimal for Silero VAD processing
- Used 16kHz sample rate as it balances quality and processing requirements
- Implemented with a buffer management system to ensure enough samples are collected for VAD processing
- Added proper resource cleanup with context managers and destructors

**Technical Challenges Solved:**
- Handling audio buffer management to match VAD requirements
- Graceful handling of microphone selection and device enumeration
- Error handling for audio stream failures

### 2. VADDetector (vad/vad_detector.py)

**Implementation Details:**
- Uses Silero VAD deep learning model via PyTorch
- Detects speech with configurable threshold (default: 0.5)
- Returns both binary speech detection and confidence scores
- Handles audio format conversion between bytes and tensors

**Design Decisions:**
- Silero VAD chosen over webrtcvad due to better accuracy and more consistent results
- Implemented requirement for specific sample sizes (512 samples for 16kHz)
- Added fallback mechanisms for different model return types
- Tracks speech probability for UI display

**Technical Challenges Solved:**
- Handling various return types from the Silero VAD model
- Processing audio in correct chunk sizes required by Silero
- Robust error handling for model prediction failures
- Converting between audio formats (bytes to NumPy arrays to PyTorch tensors)

### 3. Main Application (main.py)

**Implementation Details:**
- Integrates AudioStream and VADDetector components
- Provides a visual interface with Rich library
- Displays real-time speech detection status and statistics
- Handles user input for microphone selection

**Design Decisions:**
- Rich library used for enhanced terminal UI
- Live updating display with speech statistics
- Visual indicators for speech detection status

## Implementation Timeline

### Phase 1: Audio Capture Setup
- Implemented AudioStream class with PyAudio
- Added microphone selection capability
- Configured for 16-bit mono PCM at 16kHz
- Added proper resource management

### Phase 2: Initial VAD Implementation
- Initially implemented using webrtcvad
- Found limitations with accuracy and consistency
- Switched to Silero VAD for better performance

### Phase 3: Silero VAD Integration
- Implemented VADDetector class using Silero VAD
- Added audio format conversion between bytes and tensors
- Configured for optimal detection with 16kHz audio

### Phase 4: Buffer Management
- Added buffer system to ensure enough samples for VAD
- Implemented sliding window approach for continuous detection
- Fixed issues with sample size requirements (512 samples for 16kHz)

### Phase 5: Error Handling and Robustness
- Added comprehensive error handling
- Implemented fallback mechanisms for model failures
- Added type checking for various return types from VAD model

### Phase 6: UI Improvements
- Enhanced terminal UI with Rich library
- Added real-time statistics display
- Implemented visual indicators for speech detection

## Technical Decisions

### Audio Format
- **Sample Rate**: 16kHz (standard for speech processing)
- **Format**: 16-bit mono PCM (PyAudio paInt16)
- **Chunk Size**: 100ms (balances latency and accuracy for VAD)

### VAD Model
- **Model**: Silero VAD via PyTorch
- **Threshold**: Default 0.5 (configurable)
- **Sample Requirements**: 512 samples for 16kHz audio

### Error Handling Strategy
- Graceful degradation when components fail
- Detailed error reporting with Rich console
- Automatic retry and fallback mechanisms

## Lessons Learned

1. **Audio Processing Challenges**:
   - Audio buffer management is critical for VAD accuracy
   - Sample rate and chunk size significantly impact detection quality

2. **Model Integration**:
   - Deep learning models can have inconsistent return types
   - Proper type checking and fallbacks are essential

3. **Resource Management**:
   - Audio resources require careful cleanup
   - Context managers are effective for resource lifecycle management

4. **Error Handling**:
   - Comprehensive error handling improves system reliability
   - Graceful degradation is important for audio processing systems

## Future Improvements

1. **Audio Processing**:
   - Add noise reduction preprocessing
   - Implement automatic gain control
   - Support for additional audio formats

2. **VAD Enhancements**:
   - Add speaker diarization (who is speaking)
   - Implement adaptive thresholding based on background noise
   - Pre-trained model customization for specific environments

3. **Performance Optimizations**:
   - Reduce CPU usage with more efficient buffer management
   - GPU acceleration options for VAD processing
   - Optimize for low-power devices

4. **User Interface**:
   - Add visualizations of audio waveform
   - Implement speech segment recording
   - Add configuration UI for VAD parameters

## Technical References

- **Silero VAD**: https://github.com/snakers4/silero-vad
- **PyAudio**: http://people.csail.mit.edu/hubert/pyaudio/
- **Rich**: https://github.com/Textualize/rich 