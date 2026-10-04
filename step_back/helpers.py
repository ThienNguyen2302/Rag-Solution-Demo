def parse_step_back_question(response: str) -> str:
    question = response.strip()

    if question.startswith("```") and question.endswith("```"):
        question = question[3:-3].strip()
        if question.startswith("text"):
            question = question[4:].strip()

    if not question:
        raise ValueError("The step-back response must contain a question")

    return question
