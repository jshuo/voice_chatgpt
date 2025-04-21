import threading
import os
import queue
import logging
import time
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
from chat_gpt import ChatGPTAssistant

# Ensure the logging configuration is set up correctly
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log format
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler("voice_chat.log", mode="w")  # Overwrite logs in the file named 'voice_chat.log'
    ]
)

class ChatGPTAssistantManager:
    def __init__(self, keyword="computer"):
        self.keyword = keyword
        self.keyword_detected = threading.Event()
        self.response_queue = queue.Queue()
        self.assistant = ChatGPTAssistant(prompt_file="prompt.txt")  # Initialize ChatGPTAssistant

    def detect_keyword(self):
        """Thread for detecting the wake word."""
        porcupine = None
        pa = None
        audio_stream = None
        access_key = os.getenv("PORCUPINE")
        try:
            logging.info("Initializing Porcupine...")
            porcupine = pvporcupine.create(access_key=access_key, keywords=[self.keyword])
            logging.info("Porcupine initialized successfully.")
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length
            )
            logging.info("Audio stream opened successfully.")
            logging.info("Listening for keyword...")
            while True:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                if porcupine.process(pcm) >= 0:
                    logging.info("Keyword detected!")
                    self.keyword_detected.set()
        except Exception as e:
            logging.error(f"Error in detect_keyword: {e}", exc_info=True)
        finally:
            if audio_stream:
                audio_stream.close()
            if pa:
                pa.terminate()
            if porcupine:
                porcupine.delete()

    def recognize_and_process_speech(self):
        """Thread for recognizing speech and processing responses."""
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        while True:
            self.keyword_detected.wait()  # Wait for keyword detection
            try:
                with mic as source:
                    logging.info("Listening for speech...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = recognizer.listen(source, timeout=10)
                    logging.info("Speech captured, recognizing...")
                    text = recognizer.recognize_google(audio)
                    logging.info(f"Recognized speech: {text}")
                    self.response_queue.put(text)  # Add recognized text to the queue
            except Exception as e:
                logging.error(f"Error in recognize_and_process_speech: {e}", exc_info=True)
            finally:
                self.keyword_detected.clear()  # Clear the event after processing

    def process_responses(self):
        """Thread for processing responses from the queue."""
        while True:
            try:
                text = self.response_queue.get(timeout=5)  # Wait for a response
                logging.info(f"Processing response for: {text}")
                reply = self.assistant.get_reply(text)
                logging.info(f"Reply: {reply}")
                self.assistant.text_to_speech(reply)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in process_responses: {e}", exc_info=True)

    def start(self):
        """Start the assistant."""
        threading.Thread(target=self.detect_keyword, daemon=True).start()
        threading.Thread(target=self.recognize_and_process_speech, daemon=True).start()
        threading.Thread(target=self.process_responses, daemon=True).start()
        while True:
            time.sleep(1)  # Keep the main thread alive

if __name__ == "__main__":
    manager = ChatGPTAssistantManager()
    manager.start()

