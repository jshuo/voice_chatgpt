import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
from dotenv import load_dotenv
import os
import chat_gpt
import threading
import logging
import time

# Configure logging to capture exceptions and debug information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

KEYWORD = "computer"  # You can create custom keywords. See the documentation at picovoice.ai for more information

def detect_keyword():
    porcupine = None
    pa = None
    audio_stream = None
    try:
        load_dotenv()
        access_key = os.getenv('PORCUPINE')
        porcupine = pvporcupine.create(access_key=access_key, keywords=[KEYWORD])
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length)
        print("Listening for keyword...")
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Keyword detected!")
                return
    except Exception as e:
        logging.error(f"Error in detect_keyword: {e}", exc_info=True)
    finally:
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
            pa.terminate()
        if porcupine is not None:
            porcupine.delete()


def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            print("Please speak...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=30)  # Removed phrase_time_limit to allow uninterrupted speech capture
            print("Audio captured.")
    except sr.WaitTimeoutError:
        print("Listening timed out while waiting for phrase to start.")
        return "Listening timed out"
    except Exception as e:
        print(f"An error occurred while capturing audio: {e}")
        return "Error capturing audio"

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return "Could not understand audio"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return "Error from the Google Speech Recognition service"

keyword_detected = threading.Event()

def keyword_detection_thread():
    while True:
        try:
            detect_keyword()
            if not keyword_detected.is_set():
                print("Keyword detected, setting flag.")
                keyword_detected.set()

                # Wait until the speech thread clears it before detecting again
                while keyword_detected.is_set():
                    time.sleep(1)  # avoid tight loop
        except Exception as e:
            logging.error(f"Error in keyword_detection_thread: {e}", exc_info=True)


def speech_recognition_thread():
    while True:
        keyword_detected.wait()  # Wait until keyword is detected
        print("Keyword detected, processing speech...")
        keyword_detected.clear()

        try:
            speech = recognize_speech()
            response = chat_gpt.get_reply(speech)
            chat_gpt.text_to_speech(response)
        except Exception as e:
            logging.error(f"Error in speech_recognition_thread: {e}", exc_info=True)
        finally:
            keyword_detected.clear()
# Create threads for keyword detection and speech recognition
keyword_thread = threading.Thread(target=keyword_detection_thread, daemon=True)
speech_thread = threading.Thread(target=speech_recognition_thread, daemon=True)

# Start the threads
keyword_thread.start()
speech_thread.start()

# Keep the main thread alive
keyword_thread.join()
speech_thread.join()