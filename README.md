# Speach to note

Lets you speek to capture notes. Recordings are transcribed using faster-whisper, and summarized using LLAMA.
All self hosted to ensure privacy.

## Folder structure

Required folder structure

- speech-to-note
  -- models
  -- notes
  -- recordings
  -- templates

## LLAMA

Download llama using `download_model.py`
Place that in the folder `models`

## Setup Python

Download and install python.

In the project directory

Create a virtual environment:
`python3 -m venv myenv`

Activate virtual environment:
`source myenv/bin/activate`

## Dependencies

Install required dependencies:
`pip install -r requirements.txt`
