"""
Speech to Note: A fully offline self-hosted service for speech-to-text and summarization
using faster-whisper for Dutch transcription and Llama for high-quality summarization

Requirements:
- Python 3.8+
- PyAudio (for audio recording)
- faster-whisper (for offline speech-to-text)
- llama-cpp-python (for offline summarization)
- Flask (for web interface)

Install dependencies with:
pip install pyaudio faster-whisper llama-cpp-python flask
"""

import os
import datetime
import json
import wave
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import nltk
from flask import Flask, render_template, request, jsonify
import re
import platform
import subprocess
from llama_cpp import Llama

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Create directories for saving files
if not os.path.exists("recordings"):
    os.makedirs("recordings")
if not os.path.exists("notes"):
    os.makedirs("notes")
if not os.path.exists("models"):
    os.makedirs("models")

# Initialize NLTK resources
nltk.download('punkt', quiet=True)

# Initialize the sentence tokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
sentence_tokenizer = PunktSentenceTokenizer()

# Define the path to the Llama model
# You'll need to download a suitable model and place it in the models directory
# For example: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main
# A good starting point is the 4-bit quantized version: llama-2-7b-chat.Q4_K_M.gguf
LLAMA_MODEL_PATH = os.path.join("models", "llama-2-13b-chat.Q4_K_M.gguf")  # Adjust filename as needed

# Initialize Llama model
print("Loading Llama model... (this may take a moment)")
try:
    # Check if model exists
    if not os.path.exists(LLAMA_MODEL_PATH):
        print(f"⚠️ Llama model not found at {LLAMA_MODEL_PATH}")
        print("Please download a Llama model (GGUF format) and place it in the models directory")
        print("See: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main")
        print("Continuing without summarization capability...")
        llama_model = None
    else:
        # Initialize the model with conservative memory settings
        llama_model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=2048,         # Context window size
            n_batch=512,        # Batch size for prompt processing
            n_threads=4,        # Number of CPU threads to use
            n_gpu_layers=0      # Set higher (e.g., 32) if you have a GPU
        )
        
        print("Llama model loaded successfully")
except Exception as e:
    print(f"Error loading Llama model: {e}")
    print("Continuing without summarization capability...")
    llama_model = None

# Initialize faster-whisper model
print("Loading Whisper model... (this may take a moment on first run)")
try:
    # Use the "medium" model for better Dutch language recognition
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Make sure you've installed faster-whisper correctly with: pip install faster-whisper")

def llama_summarize(transcript, num_points=3):
    """Generate a summary using Llama model"""
    if llama_model is None:
        return "• Llama model not loaded. Cannot generate summary.\n• Please check the full transcription."
    
    # Construct a prompt for the Llama model that works well with instruction-tuned models
    prompt = f"""[INST] Je bent een behulpzame assistent die transcripties in het Nederlands samenvat.

Maak een samenvatting van de volgende transcriptie in minimaal {num_points} duidelijke en beknopte hoofdpunten.
Elk punt moet een belangrijk idee of concept uit de transcriptie weergeven.
Presenteer elk punt als een bullet point startend met "•".

Hier is de transcriptie:

{transcript}

Geef alleen de samenvatting. De samenvatting moet alle belangrijke concepten en actiepunten uit de transcriptie bevatten. Niet meer en niet minder.[/INST]
"""

    try:
        # Generate summary with appropriate parameters for quality output
        response = llama_model(
            prompt,
            max_tokens=1024,
            temperature=0.1,   # Lower temperature for more focused output
            top_p=0.9,         # Slightly constrained sampling for coherence
            repeat_penalty=1.1 # Discourage repetition
        )
        
        # Extract the summary text
        summary_text = response["choices"][0]["text"].strip()
        
        # If the summary doesn't contain bullet points, format it
        if "•" not in summary_text:
            # Split by newlines or numbers to identify separate points
            points = re.split(r'\n+|\d+\.', summary_text)
            points = [p.strip() for p in points if p.strip()]
            
            # Format as bullet points
            formatted_points = []
            for point in points[:num_points]:
                if not point.startswith("•"):
                    point = "• " + point
                if not any(point.rstrip().endswith(p) for p in ['.', '!', '?']):
                    point = point + "."
                formatted_points.append(point)
            
            summary_text = "\n".join(formatted_points)
        
        return summary_text
    except Exception as e:
        print(f"Error generating summary with Llama: {e}")
        return "• Error generating summary.\n• Please check the full transcription."

def format_transcript(segments):
    """
    Format transcription with paragraph breaks at natural pauses and topic changes.
    
    Args:
        segments: The segments returned by faster-whisper transcription
        
    Returns:
        Formatted transcript text with appropriate paragraph breaks
    """
    # Constants for pause detection and paragraph formation
    LONG_PAUSE_THRESHOLD = 1.0  # seconds - consider a pause over this length as paragraph break
    MEDIUM_PAUSE_THRESHOLD = 0.7  # seconds - consider a pause as potential break if other conditions are met
    MIN_PARAGRAPH_LENGTH = 100  # minimum character length for paragraphs
    
    current_paragraph = []
    paragraphs = []
    last_end_time = 0
    paragraph_text_length = 0
    
    # Process all segments to collect text and identify pauses
    for segment in segments:
        # Calculate pause since last segment
        pause_duration = segment.start - last_end_time if last_end_time > 0 else 0
        text = segment.text.strip()
        
        # Skip empty segments
        if not text:
            continue

        # Detect sentence endings (potential break points)
        ends_with_sentence = any(text.rstrip().endswith(p) for p in ['.', '!', '?'])
        
        # Check for conditions to create a new paragraph
        new_paragraph = False
        
        # Long pause is a strong indicator for paragraph break
        if pause_duration > LONG_PAUSE_THRESHOLD and paragraph_text_length > MIN_PARAGRAPH_LENGTH:
            new_paragraph = True
            
        # Medium pause + sentence ending suggests a natural break
        elif pause_duration > MEDIUM_PAUSE_THRESHOLD and ends_with_sentence and paragraph_text_length > MIN_PARAGRAPH_LENGTH:
            new_paragraph = True
        
        # Create a new paragraph if conditions are met
        if new_paragraph and current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
            paragraph_text_length = 0
        
        # Add text to current paragraph
        current_paragraph.append(text)
        paragraph_text_length += len(text)
        last_end_time = segment.end
    
    # Add the last paragraph if not empty
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Join paragraphs with double newlines for proper separation
    return '\n\n'.join(paragraphs)

# Transcription function
def transcribe_audio(audio_file_path):
    """Convert speech to text using faster-whisper with enhanced formatting"""
    try:
        # Transcribe the audio file
        segments, info = whisper_model.transcribe(
            audio_file_path,
            language="nl",  # Dutch language
            beam_size=5,    # Higher values provide better results but are slower
            vad_filter=True, # Filter out non-speech parts
            word_timestamps=True # Get timestamps for each word
        )
        
        # Format the transcript with paragraph breaks
        # First collect segments (need to do this since segments is a generator)
        segment_list = list(segments)
        
        # Format transcript with improved paragraph structure
        formatted_transcript = format_transcript(segment_list)
        
        return formatted_transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error transcribing audio: {str(e)}"

# Save note function
def save_note(transcription, summary):
    """Save the transcription and summary as a note"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transcription
    with open(f"notes/{timestamp}_transcription.txt", "w") as f:
        f.write(transcription)
    
    # Save summary
    with open(f"notes/{timestamp}_summary.txt", "w") as f:
        f.write(summary)
    
    return timestamp

# Audio recorder class
class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.sample_rate = 16000  # 16kHz for Whisper
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.frames = []
        self.stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback
        )
        self.stream.start_stream()
        print("Recording started...")

    def _callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def stop_recording(self):
        """Stop recording and save the audio file"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Generate timestamp and filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/{timestamp}.wav"
        
        # Save the recorded audio to a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print(f"Recording saved as {filename}")
        return filename

# Create audio recorder instance
audio_recorder = AudioRecorder()

# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/start_record', methods=['POST'])
def start_record():
    """Start recording audio"""
    if audio_recorder.is_recording:
        return jsonify({"success": False, "message": "Already recording"})
    
    audio_recorder.start_recording()
    return jsonify({"success": True, "message": "Recording started"})

@app.route('/stop_record', methods=['POST'])
def stop_record():
    """Stop recording and process the audio"""
    if not audio_recorder.is_recording:
        return jsonify({"success": False, "message": "Not recording"})
    
    # Stop recording and get the filename
    audio_file = audio_recorder.stop_recording()
    if not audio_file:
        return jsonify({"success": False, "message": "Failed to save recording"})
    
    # Process the audio
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file)
    
    print("Generating summary with Llama...")
    try:
        summary = llama_summarize(transcription, num_points=5)
    except Exception as e:
        print(f"Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        summary = "• Could not generate summary.\n• Please check the full transcription."
    
    # Save as note
    note_id = save_note(transcription, summary)
    
    return jsonify({
        "success": True,
        "transcription": transcription,
        "summary": summary,
        "note_id": note_id
    })

@app.route('/notes')
def list_notes():
    """List all saved notes"""
    notes = []
    for file in os.listdir("notes"):
        if file.endswith("_summary.txt"):
            note_id = file.split("_")[0]
            
            # Read the summary
            with open(f"notes/{file}", "r") as f:
                summary = f.read()
            
            # Handle different date formats safely
            try:
                if "_" in note_id:
                    # Format with time: 20250410_123045
                    date_str = datetime.datetime.strptime(note_id, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Format without time: 20250410
                    date_str = datetime.datetime.strptime(note_id, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                # Fallback for any other format
                date_str = note_id
            
            notes.append({
                "id": note_id,
                "date": date_str,
                "summary": summary[:100] + "..." if len(summary) > 100 else summary
            })
    
    return jsonify({"notes": sorted(notes, key=lambda x: x["id"], reverse=True)})

@app.route('/note/<note_id>')
def get_note(note_id):
    """Get a specific note"""
    try:
        # Check if files exist and handle different possible formats
        transcription_file = None
        summary_file = None
        
        # Check direct match
        if os.path.exists(f"notes/{note_id}_transcription.txt"):
            transcription_file = f"notes/{note_id}_transcription.txt"
            summary_file = f"notes/{note_id}_summary.txt"
        
        # Check if note_id is only a date without time
        elif "_" in note_id:
            date_part = note_id.split("_")[0]
            # Search for files starting with this date
            for file in os.listdir("notes"):
                if file.startswith(date_part) and file.endswith("_transcription.txt"):
                    transcription_file = f"notes/{file}"
                    summary_file = f"notes/{file.replace('_transcription.txt', '_summary.txt')}"
                    break
        
        # If files not found, return error
        if not transcription_file or not os.path.exists(transcription_file) or \
           not summary_file or not os.path.exists(summary_file):
            return jsonify({
                "success": False,
                "error": f"Note files not found: {transcription_file}, {summary_file}"
            })
        
        # Read files
        with open(transcription_file, "r") as f:
            transcription = f.read()
        
        with open(summary_file, "r") as f:
            summary = f.read()
        
        # Get absolute paths
        transcription_path = os.path.abspath(transcription_file)
        summary_path = os.path.abspath(summary_file)
        
        return jsonify({
            "success": True,
            "transcription": transcription,
            "summary": summary,
            "transcription_path": transcription_path,
            "summary_path": summary_path
        })
    except Exception as e:
        print(f"Error getting note {note_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error accessing note: {str(e)}"
        })

@app.route('/open_file')
def open_file():
    """Open a note file in the system's default application"""
    file_path = request.args.get('path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"success": False, "message": "File not found"})
    
    try:
        # Open file with default application based on operating system
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.run(['open', file_path])
        elif system == 'Windows':
            os.startfile(file_path)
        else:  # Linux and others
            subprocess.run(['xdg-open', file_path])
            
        return jsonify({"success": True, "message": "File opened"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error opening file: {str(e)}"})

@app.route('/check_model')
def check_model():
    """Check if the Llama model is available and loaded"""
    model_exists = os.path.exists(LLAMA_MODEL_PATH)
    model_loaded = llama_model is not None
    
    return jsonify({
        "model_exists": model_exists,
        "model_loaded": model_loaded,
        "model_path": LLAMA_MODEL_PATH,
        "message": "Llama model is loaded and ready" if model_loaded else 
                  "Llama model file exists but could not be loaded" if model_exists else
                  "Llama model file not found"
    })

# Run the app
if __name__ == "__main__":
    print("Starting Speech-to-Note service...")
    print("Access the web interface at http://localhost:5000")
    app.run(debug=True)