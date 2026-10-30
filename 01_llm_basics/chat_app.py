import os
import time

import ollama
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are a helpful AI tutor. "
    "Explain technical concepts clearly, "
    "use Python examples when useful, "
    "and adapt your explanations to the learner's level."
)


def ask_llm(messages: list[dict]) -> dict:
    """Send conversation messages to the local LLM."""

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "temperature": TEMPERATURE
        },
    )

    return response


def main():
    # Keep the conversation history in memory.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print(f"Local LLM Chat — Model: {MODEL}")
    print("Ask a question, enter /clear to reset, or /exit to quit.")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() == "/exit":
            print("Goodbye!")
            break

        if question.lower() == "/clear":
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]
            print("Conversation history cleared.")
            continue

        # Add the user's message to the conversation.
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        start_time = time.perf_counter()

        try:
            response = ask_llm(messages)

            elapsed = time.perf_counter() - start_time
            answer = response["message"]["content"]

            # Save the assistant's response for future context.
            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            print(f"\nAssistant: {answer}")
            print(f"\nResponse time: {elapsed:.2f} seconds")

            # Ollama may provide these usage counters.
            prompt_tokens = response.get("prompt_eval_count")
            generated_tokens = response.get("eval_count")

            if prompt_tokens is not None:
                print(f"Prompt tokens: {prompt_tokens}")

            if generated_tokens is not None:
                print(f"Generated tokens: {generated_tokens}")

        except ollama.ResponseError as error:
            # Remove the unanswered question from the history.
            messages.pop()
            print(f"Ollama model error: {error}")

        except Exception as error:
            messages.pop()
            print(f"Could not complete the request: {error}")
            print("Check that Ollama is running and the model is available.")


if __name__ == "__main__":
    main()