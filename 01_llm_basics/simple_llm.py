import os
import time

import ollama
from dotenv import load_dotenv

# Load configuration from .env
load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def ask_llm(question: str):
    """Send a question to a locally running LLM."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI tutor. "
                    "Explain technical concepts clearly "
                    "and provide Python examples when useful."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response


def main():
    question = input("Ask your local LLM a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    start_time = time.perf_counter()

    try:
        response = ask_llm(question)

        elapsed = time.perf_counter() - start_time

        print("\nLLM Response:")
        print(response["message"]["content"])

        print(f"\nResponse time: {elapsed:.2f} seconds")
        print(f"Model: {MODEL}")

        print("\nPrompt tokens:", response.get("prompt_eval_count"))
        print("Generated tokens:", response.get("eval_count"))

    except ollama.ResponseError as error:
        print(f"Ollama model error: {error}")

    except Exception as error:
        print(f"Could not connect to Ollama: {error}")
        print("Check that Ollama is installed and running.")


if __name__ == "__main__":
    main()
