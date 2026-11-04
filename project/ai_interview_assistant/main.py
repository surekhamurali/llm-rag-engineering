import time

import ollama

from llm import ask_llm, MODEL
from prompts import (
    ROLES,
    DIFFICULTIES,
    get_question_prompt,
    get_evaluation_prompt,
)


def display_options(title, options):
    """Display available choices."""

    print(f"\n{title}")

    for key, value in options.items():
        print(f"{key}. {value}")


def get_user_choice(options):
    """Get a valid choice from the user."""

    while True:
        choice = input("\nEnter your choice: ").strip()

        if choice in options:
            return options[choice]

        print("Invalid choice. Please try again.")


def generate_question(role, difficulty, previous_questions):
    """Generate a technical interview question."""

    prompt = get_question_prompt(
        role,
        difficulty,
        previous_questions,
    )

    result = ask_llm(prompt)

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    question = result.get("question")
    expected_topics = result.get("expected_topics")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Invalid question returned by the model.")

    if not isinstance(expected_topics, list) or not all(
        isinstance(topic, str) for topic in expected_topics
    ):
        raise ValueError("Invalid expected_topics returned by the model.")

    return result


def evaluate_answer(role, difficulty, question, answer):
    """Evaluate the candidate's answer."""

    prompt = get_evaluation_prompt(
        role,
        difficulty,
        question,
        answer,
    )

    result = ask_llm(prompt)

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    score = result.get("score")

    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError("The score must be an integer.")

    if not 1 <= score <= 5:
        raise ValueError("The score must be between 1 and 5.")

    for field in ("strengths", "areas_to_improve"):
        value = result.get(field)

        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"Invalid {field} returned by the model.")

    for field in ("feedback", "suggested_answer"):
        if not isinstance(result.get(field), str):
            raise ValueError(f"Invalid {field} returned by the model.")

    return result


def display_feedback(result):
    """Display evaluation results."""

    print("\n" + "=" * 50)
    print("INTERVIEW FEEDBACK")
    print("=" * 50)

    print(f"\nScore: {result['score']}/5")

    print("\nStrengths:")

    for strength in result["strengths"]:
        print(f"- {strength}")

    print("\nAreas to improve:")

    for improvement in result["areas_to_improve"]:
        print(f"- {improvement}")

    print("\nOverall feedback:")
    print(result["feedback"])

    print("\nSuggested answer:")
    print(result["suggested_answer"])

    print("=" * 50)


def display_final_report(scores):
    """Display the final interview report."""

    if not scores:
        print("\nNo answers were evaluated.")
        return

    total_score = sum(scores)
    average_score = total_score / len(scores)

    print("\n" + "=" * 50)
    print("FINAL INTERVIEW REPORT")
    print("=" * 50)

    print(f"Questions answered: {len(scores)}")
    print(f"Total score: {total_score}/{len(scores) * 5}")
    print(f"Average score: {average_score:.2f}/5")

    if average_score >= 4:
        print("Performance: Strong")
    elif average_score >= 3:
        print("Performance: Good, with room for improvement")
    else:
        print("Performance: Needs more preparation")

    print("=" * 50)


def main():

    print("\n" + "=" * 50)
    print("AI INTERVIEW ASSISTANT")
    print("=" * 50)

    print(f"Model: {MODEL}")

    # Step 1: Select interview role.
    display_options("Select your interview role:", ROLES)
    role = get_user_choice(ROLES)

    # Step 2: Select difficulty.
    display_options("Select difficulty:", DIFFICULTIES)
    difficulty = get_user_choice(DIFFICULTIES)

    # Step 3: Select number of questions.
    while True:
        try:
            total_questions = int(
                input("\nNumber of questions (1-10): ")
            )

            if 1 <= total_questions <= 10:
                break

            print("Enter a number between 1 and 10.")

        except ValueError:
            print("Please enter a valid number.")

    previous_questions = []
    scores = []

    print("\nStarting your AI interview...")
    print(f"Role: {role}")
    print(f"Difficulty: {difficulty}")

    for question_number in range(1, total_questions + 1):

        print("\n" + "=" * 50)
        print(f"QUESTION {question_number}/{total_questions}")
        print("=" * 50)

        try:
            # Generate question.
            question_result = generate_question(
                role,
                difficulty,
                previous_questions,
            )

            question = question_result["question"]

            previous_questions.append(question)

            print(f"\nInterviewer: {question}")

            if question_result["expected_topics"]:
                print("\nExpected concepts:")
                for topic in question_result["expected_topics"]:
                    print(f"- {topic}")

            answer = input("\nYour answer: ").strip()

            if not answer:
                print("Empty answer. Moving to the next question.")
                continue

            if answer.lower() == "/exit":
                print("Interview stopped by candidate.")
                break

            # Evaluate answer.
            start_time = time.perf_counter()

            evaluation = evaluate_answer(
                role,
                difficulty,
                question,
                answer,
            )

            elapsed = time.perf_counter() - start_time

            scores.append(evaluation["score"])

            display_feedback(evaluation)

            print(f"\nEvaluation time: {elapsed:.2f} seconds")

        except (ollama.ResponseError, ValueError, KeyError) as error:
            print(f"\nCould not process this question: {error}")

        except Exception as error:
            print(f"\nUnexpected error: {error}")

    display_final_report(scores)


if __name__ == "__main__":
    main()