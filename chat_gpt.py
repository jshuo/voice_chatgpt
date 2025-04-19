import os
from pathlib import Path
from dotenv import load_dotenv
import openai


def get_reply(text_input):
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": f"You are an assistant. {PROMPT}"
            },
            {
                "role": "user",
                "content": text_input
            }
        ]
    )
    raw_response = response.choices[0].message.content
    return raw_response


def load_prompt(prompt_file):
    with open(f'./{prompt_file}') as f:
        return f.read()


def text_to_speech(text):
    speech_file_path = Path(__file__).parent / "output.mp3"
    response = openai.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )

    response.stream_to_file(speech_file_path)
    os.system("mpg321 output.mp3")


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

PROMPT = load_prompt('prompt.txt')