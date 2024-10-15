# rag_system/processing/prompt_constructor.py

from typing import List

def construct_extrapolation_prompt(entity1: str, relation: str, entity2: str, known_facts: List[str]) -> str:
    """
    Construct a prompt for the LLM to extrapolate the veracity of the relation.

    :param entity1: The first entity.
    :param relation: The relation to assess.
    :param entity2: The second entity.
    :param known_facts: A list of known facts related to the entities.
    :return: The constructed prompt string.
    """
    facts_str = "\n".join(f"- {fact}" for fact in known_facts)
    prompt = (
        f"Based on the following known facts:\n"
        f"{facts_str}\n"
        f"\nAssess the likelihood that '{entity1}' and '{entity2}' have the relation '{relation}'. "
        f"Provide a detailed explanation and estimate the confidence level."
    )
    return prompt
