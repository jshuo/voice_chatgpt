import os
from pathlib import Path
from dotenv import load_dotenv
import openai


class ChatGPTAssistant:
    def __init__(self, prompt_file: str, api_key_env: str = 'OPENAI_API_KEY'):
        """Initialize the assistant with a prompt and API key."""
        load_dotenv()
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        openai.api_key = self.api_key

        self.prompt = self._load_prompt(prompt_file)

    def _load_prompt(self, prompt_file: str) -> str:
        """Load the prompt from a file."""
        try:
            with open(prompt_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file '{prompt_file}' not found.")

    def get_reply(self, text_input: str) -> str:
        """Get a reply from the OpenAI API."""
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": f"You are an assistant. {self.prompt}"},
                    {"role": "user", "content": text_input}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error in get_reply: {e}")

    def text_to_speech(self, text: str, output_file: str = "output.mp3", voice: str = "nova"):
        """Convert text to speech and save it to a file."""
        try:
            speech_file_path = Path(__file__).parent / output_file
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            response.stream_to_file(speech_file_path)
            os.system(f"mpg321 {speech_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error in text_to_speech: {e}")