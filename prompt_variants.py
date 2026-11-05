
def generate_prompt_variants(prompt):
    prompt = prompt.strip()

    if not prompt:
        return {}

    return {
        "Original": prompt,
        "Instruction Added": (
            "Please answer the following question clearly: " + prompt
        ),
        "Direct Instruction": (
            "Provide a direct and accurate answer to this question: " + prompt
        ),
        "Detailed Instruction": (
            "Read carefully and explain your answer step by step: " + prompt
        ),
        "Concise Instruction": (
            "Answer briefly and precisely: " + prompt
        )
    }
