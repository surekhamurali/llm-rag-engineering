import os
import time

import ollama
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def ask_llm(system_prompt: str, user_prompt: str) -> tuple[str, float]:
    """Send a prompt to Ollama and return the answer and elapsed time."""

    start_time = time.perf_counter()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.3,
        },
    )

    elapsed = time.perf_counter() - start_time

    return response["message"]["content"], elapsed


def main():
    question = input("Enter a technical question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # TODO 1: Define a simple system prompt.
    zero_shot_system = (
        "You are a helpful AI tutor. "
        "Explain technical concepts in simple language. "
        "Use Python examples whenever useful."
    )

    # TODO 2: Write a few-shot prompt containing at least
    # two input-output examples and the user's question.
    few_shot_prompt = """
    You are a helpful Python programming tutor.

    Example 1:
    Question: What is a Python list?
    Answer: A list stores multiple items in a single variable.

    Example 2:
    Question: What is a Python dictionary?
    Answer: A dictionary stores data in key-value pairs.

    Example 3:
    Question: What is a Python tuple?
    Answer: A tuple is an ordered collection that cannot be modified.

    Now answer the following question using the same style.

    Question: """ + question

    # TODO 3: Define a role-based system prompt.
    role_system = role_system = (
        "You are a senior Python software engineer "
        "with 10 years of professional experience. "
        "You are mentoring a junior Python developer. "
        "Explain concepts step by step. "
        "Include practical Python code examples. "
        "Explain real-world applications and best practices."
    )

    experiments = [
        ("Zero-shot", zero_shot_system, question),
        ("Few-shot", zero_shot_system, few_shot_prompt),
        ("Role prompting", role_system, question),
    ]

    for name, system_prompt, user_prompt in experiments:
        print(f"\n{'=' * 15} {name} {'=' * 15}")

        try:
            answer, elapsed = ask_llm(
                system_prompt,
                user_prompt,
            )

            print(answer)
            print(f"\nResponse time: {elapsed:.2f} seconds")

        except ollama.ResponseError as error:
            print(f"Ollama model error: {error}")

        except Exception as error:
            print(f"Request failed: {error}")


if __name__ == "__main__":
    main()