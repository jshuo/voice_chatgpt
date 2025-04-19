import os
import numpy as np
import pyaudio
import time
import logging
import threading
import queue
import openai
import pvporcupine
import struct
from dotenv import load_dotenv  # Import dotenv to load environment variables
from pathlib import Path
import speech_recognition as sr

# Load environment variables from .env file
load_dotenv()

# Retrieve keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
access_key = os.getenv("PORCUPINE")

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] remains unchanged as it points to a file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-service.json"

# Set up the sounddevice stream
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
CHANNELS = 1


# Initialize variables for the timeout mechanism
timeout_seconds = 5
q = queue.Queue()
stop_recording = threading.Event()


def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.int16)
    q.put(data)
    return (in_data, pyaudio.paContinue)



# Initialize Porcupine
keyword_file_path = "computer_raspberry-pi.ppn"

KEYWORD = "computer"  # You can create custom keywords. See the documentation at picovoice.ai for more information


porcupine = pvporcupine.create(access_key=access_key, keywords=[KEYWORD])

def wake_words_detect():
    """Detects the wake word using Porcupine."""
    while True:
        pa = None
        audio_stream = None
        try:
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length
            )
            print("Listening for keyword...")

            while True:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Keyword detected!")
                    return True
        except Exception as e:
            logging.error(f"Error in wake_words_detect: {e}", exc_info=True)
            return False
        finally:
            if audio_stream:
                audio_stream.close()
            if pa:
                pa.terminate()


def process_responses(response_queue):
    """Processes responses from the queue and interacts with OpenAI API."""
    transcript = ''
    logging.info("process_responses started.")  # Log when the function starts

    while not stop_recording.is_set():
        try:
            logging.debug("Waiting for response from queue...")
            response = response_queue.get(timeout=timeout_seconds)
            logging.debug(f"Received response from queue: {response}")
            if response:
                transcript += response + " "
        except queue.Empty:
            logging.warning("Queue is empty. Breaking out of the loop.")
            break

    if transcript:
        messages = [
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": transcript.strip()}
        ]
        try:
            logging.info("Sending transcript to OpenAI API.")
            chat = openai.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content
            logging.info("Received reply from OpenAI API.")
            print("Response: {0}".format(reply))
            text_to_speech("en-US-Studio-O", reply)
        except Exception as e:
            logging.error(f"Error in process_responses: {e}", exc_info=True)
    else:
        logging.warning("No transcript to process.")

    logging.info("process_responses finished.")


def text_to_speech(voice_name: str, text: str):
    """Converts text to speech using OpenAI's API and plays the audio."""
    try:
        speech_file_path = Path(__file__).parent / "output.mp3"
        response = openai.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )

        response.stream_to_file(speech_file_path)
        os.system("mpg321 output.mp3")
    except Exception as e:
        logging.error(f"Error in text_to_speech: {e}", exc_info=True)


def main():
    """Main function to handle the voice assistant logic."""
    global stop_recording
    count = 0
    while count < 1000:
        if wake_words_detect():
            response_queue = queue.Queue()
            processing_thread = threading.Thread(target=process_responses, args=(response_queue,))
            processing_thread.start()

            try:
                recognizer = sr.Recognizer()
                with sr.Microphone() as mic:
                    print("Please speak...")
                    recognizer.adjust_for_ambient_noise(mic, duration=1)

                    # Extend timeout and phrase time limits
                    audio = recognizer.listen(mic, timeout=60, phrase_time_limit=30)
                    print("Audio captured.")

                    # Recognize speech and handle retries for incomplete phrases
                    try:
                        text = recognizer.recognize_google(audio)
                        print("You said: " + text)
                        logging.info(f"Recognized text: {text}")  # Log recognized text
                        response_queue.put(text)
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio.")
                        logging.warning("Speech recognition failed: UnknownValueError")
                        response_queue.put("[Unrecognized audio]")
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")
                        logging.error(f"Speech recognition failed: RequestError - {e}")
                        response_queue.put("[Error in recognition]")

            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase to start.")
                logging.warning("Listening timed out.")
            except Exception as e:
                logging.error(f"Error in main speech recognition: {e}", exc_info=True)

            # Ensure the queue is populated before starting process_responses
            if not response_queue.empty():
                processing_thread.join()
            else:
                logging.warning("Queue was empty when process_responses started.")
                time.sleep(0.5)  # Additional delay to allow queue population
                processing_thread.join()

            stop_recording.clear()

        print(f"count: {count}")
        count += 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log format
        handlers=[
            logging.StreamHandler(),  # Output logs to the console
            logging.FileHandler("voice_chat.log", mode="a")  # Save logs to a file named 'voice_chat.log'
        ]
    )
    main()


