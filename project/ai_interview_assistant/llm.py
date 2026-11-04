import os
import json

import ollama
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
TEMPERATURE = 0.3


def ask_llm(prompt):
    """Send a prompt to the local LLM and return JSON data."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format="json",
        options={
            "temperature": TEMPERATURE,
        },
    )

    content = response["message"]["content"]

    return json.loads(content)