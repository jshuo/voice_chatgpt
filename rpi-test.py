import os
import numpy as np
import pyaudio
import time
import threading
import queue
import openai
import pvporcupine
import struct

from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1.types import RecognitionConfig, StreamingRecognitionConfig, StreamingRecognizeRequest
from google.cloud import texttospeech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-service.json"

# Set up the sounddevice stream
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
CHANNELS = 1

# Set up the Google Cloud Speech client
client = speech.SpeechClient()
config = RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code='en-US')

streaming_config = StreamingRecognitionConfig(
    config=config,
    interim_results=True)

# Initialize variables for the timeout mechanism
timeout_seconds = 5
q = queue.Queue()
stop_recording = threading.Event()


def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.int16)
    q.put(data)
    return (in_data, pyaudio.paContinue)

# Generator for streaming audio data to the API
def generate_audio_data():
    while not stop_recording.is_set():
        data = q.get()
        yield StreamingRecognizeRequest(audio_content=data.tobytes())

# Initialize Porcupine
keyword_file_path = "computer_raspberry-pi.ppn"
access_key = "bxcqMTBJlO5uxQpuLXZCWUb1okVHXVYlvGbNz9VeM/16d1x5O9zivg=="

porcupine = pvporcupine.create(
    access_key=access_key, keyword_paths=[keyword_file_path])

def wake_words_detect():
    # Set up PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(rate=porcupine.sample_rate,
                            channels=1,
                            format=pyaudio.paInt16,
                            input=True,
                            frames_per_buffer=porcupine.frame_length)

    # Listen for the wake word
    print("Listening for wake word...")
    while True:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        result = porcupine.process(pcm)

        if result >= 0:
            print("Wake word detected!")
            # Stop audio stream
            stream.stop_stream()
            stream.close()
            # audio.terminate()
            return True
        
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
    global transcript
    transcript =''
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

count = 0
while count < 1000:
    ww_detect = wake_words_detect()

    if (ww_detect):
        ww_detect = False
        # Start a thread for processing responses
        response_queue = queue.Queue()
        processing_thread = threading.Thread(target=process_responses)
        processing_thread.start()

        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=callback)

        ...
        stream.start_stream()

        # Start the recognition stream
        responses = client.streaming_recognize(streaming_config, generate_audio_data())

        # Forward the responses to the response_queue
        for response in responses:
            response_queue.put(response)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Wait for the processing thread to finish
        processing_thread.join()
        stop_recording.clear()
    print(f'count: {count}')
    count += 1


