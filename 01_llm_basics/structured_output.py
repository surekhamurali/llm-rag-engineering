import json
import ollama

MODEL = "llama3.2:1b"

SYSTEM_PROMPT = """
You are a technical interview assistant.

Generate one Python interview question.

Return a JSON object with exactly these fields:
- question: a string
- answer: a string
- difficulty: beginner, intermediate, or advanced
- skills: a list of strings

Return valid JSON only.
Do not include Markdown code fences.
"""


def generate_interview_question(topic: str) -> dict:
    """Generate and parse a structured interview question."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Generate a question about: {topic}",
            },
        ],
        format="json",
        options={
            "temperature": 0.2,
        },
    )

    content = response["message"]["content"]

    # Parse the model's JSON string into a Python dictionary.
    data = json.loads(content)

    # Basic schema validation.
    required_fields = {
        "question",
        "answer",
        "difficulty",
        "skills",
    }

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")

    if not required_fields.issubset(data.keys()):
        raise ValueError("The response is missing required fields.")

    if not isinstance(data["question"], str):
        raise ValueError("'question' must be a string.")

    if not isinstance(data["answer"], str):
        raise ValueError("'answer' must be a string.")

    if data["difficulty"] not in {
        "beginner",
        "intermediate",
        "advanced",
    }:
        raise ValueError("Invalid difficulty value.")

    if not isinstance(data["skills"], list) or not all(
        isinstance(skill, str) for skill in data["skills"]
    ):
        raise ValueError("'skills' must be a list of strings.")

    return data


def main():
    topic = input("Enter an interview topic: ").strip()

    if not topic:
        print("Please enter a topic.")
        return

    try:
        result = generate_interview_question(topic)

        print("\nGenerated interview question:")
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError:
        print("The model returned invalid JSON.")

    except ollama.ResponseError as error:
        print(f"Ollama error: {error}")

    except Exception as error:
        print(f"Could not generate the question: {error}")


if __name__ == "__main__":
    main()