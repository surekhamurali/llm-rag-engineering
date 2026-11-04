# Available interview roles

ROLES = {
    "1": "Python Developer",
    "2": "Computer Vision Engineer",
    "3": "AI / LLM Engineer",
}


DIFFICULTIES = {
    "1": "Beginner",
    "2": "Intermediate",
    "3": "Advanced",
}


def get_question_prompt(role, difficulty, previous_questions):
    """Create a prompt to generate an interview question."""

    previous = "\n".join(previous_questions) or "None"

    return f"""
You are an experienced technical interviewer.

Interview role: {role}
Difficulty: {difficulty}

Generate ONE technical interview question.

Requirements:
1. The question must be relevant to the selected role.
2. Match the requested difficulty.
3. Avoid repeating previous questions.
4. Prefer practical engineering questions.
5. Include the important concepts expected in a good answer.

Previous questions:
{previous}

Return valid JSON with exactly these fields:

{{
    "question": "The interview question",
    "expected_topics": [
        "topic 1",
        "topic 2"
    ]
}}

Return JSON only.
"""


def get_evaluation_prompt(role, difficulty, question, answer):
    """Create a prompt to evaluate a candidate's answer."""

    return f"""
You are a senior technical interviewer.

Evaluate the candidate's answer objectively.

Interview role: {role}
Difficulty: {difficulty}

Interview question:
{question}

Candidate answer:
{answer}

Evaluate based on:
1. Technical correctness.
2. Understanding of the concept.
3. Completeness.
4. Practical engineering knowledge.

Scoring:
1 = Incorrect or very poor.
2 = Limited understanding.
3 = Partially correct.
4 = Good answer with minor gaps.
5 = Excellent and complete answer.

Do not give a high score just because the answer sounds confident.

Return valid JSON with exactly these fields:

{{
    "score": 3,
    "strengths": [
        "Strength 1"
    ],
    "areas_to_improve": [
        "Improvement 1"
    ],
    "feedback": "Overall feedback",
    "suggested_answer": "An improved technical interview answer"
}}

The score must be an integer from 1 to 5.

Return JSON only.
"""