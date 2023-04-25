import os
import numpy as np
import sounddevice as sd
import time
import threading
import queue
import openai
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1.types import RecognitionConfig, StreamingRecognitionConfig, StreamingRecognizeRequest
from google.cloud import texttospeech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-service.json"

# Set up the sounddevice stream
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
CHANNELS = 1

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# Set up the Google Cloud Speech client
client = speech.SpeechClient()
config = RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code='en-US')

streaming_config = StreamingRecognitionConfig(
    config=config,
    interim_results=True)

# Generator for streaming audio data to the API
def generate_audio_data():
    while not stop_recording.is_set():
        data = q.get()
        yield StreamingRecognizeRequest(audio_content=data.tobytes())

# Initialize variables for the timeout mechanism
timeout_seconds = 5
q = queue.Queue()
stop_recording = threading.Event()
transcript=''

def text_to_speech(voice_name: str, text: str):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-service.json"
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3)

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = f"{voice_name}.mp3"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

# Function to process responses
def process_responses():
    while not stop_recording.is_set():
        try:
            response = response_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            print("No more responses received. Exiting...")
            messages = [{"role": "system", "content":
                "You are a intelligent assistant."}]
            # message = "where is taipei?"
            if transcript:
                messages.append(
                    {"role": "user", "content": transcript},
                )
                openai.api_key = "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"
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages
                )
                reply = chat.choices[0].message.content

                # Send text to ChatGPT.

                print("Response: {0}".format(reply))

                # Convert ChatGPT response into audio.
                text_to_speech("en-US-Studio-O", reply)

                import pygame

                pygame.init()
                pygame.mixer.init()

                pygame.mixer.music.load('en-US-Studio-O.mp3')  # replace with your MP3 file
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    # check if playback has finished
                    pygame.time.Clock().tick(10)

                pygame.quit()
            stop_recording.set()
            break

        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        print(f'Transcript: {transcript}')

# Start a thread for processing responses
response_queue = queue.Queue()
processing_thread = threading.Thread(target=process_responses)
processing_thread.start()

# Start the audio stream
with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=np.int16, blocksize=CHUNK, callback=callback):
    # Stream audio and put responses into the queue
    requests = generate_audio_data()
    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        response_queue.put(response)

# Signal the processing thread to stop
stop_recording.set()

# Wait for the processing thread to finish
processing_thread.join()


