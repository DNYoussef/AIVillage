from typing import Any


def construct_user_intent_prompt(query: str) -> str:
    """Construct a prompt for the User Intent Interpretation Agent.

    Args:
        query (str): The user's input query.

    Returns:
        str: The constructed prompt string.
    """
    prompt = (
        "You are an intent interpretation agent.\n"
        f"User Query: '{query}'\n"
        "Interpret the user's intent and categorize the query into predefined structures. "
        "Provide the intent type, relevant entities, and a confidence score."
    )
    return prompt

def construct_key_concept_extraction_prompt(text: str) -> str:
    """Construct a prompt for the Key Concept Extraction Agent.

    Args:
        text (str): Input text from which to extract key concepts.

    Returns:
        str: The constructed prompt string.
    """
    prompt = (
        "You are a key concept extraction agent.\n"
        f"Text: '{text}'\n"
        "Extract key entities and keywords using advanced NLP techniques. "
        "Provide the entities and keywords in a structured format."
    )
    return prompt

def construct_task_planning_prompt(intent: dict[str, Any], concepts: dict[str, Any]) -> str:
    """Construct a prompt for the Task Planning Agent.

    Args:
        intent (Dict[str, Any]): The interpreted user intent.
        concepts (Dict[str, Any]): The extracted key concepts.

    Returns:
        str: The constructed prompt string.
    """
    prompt = (
        "You are a task planning agent.\n"
        f"User Intent: {intent}\n"
        f"Key Concepts: {concepts}\n"
        "Based on the intent and key concepts, generate a detailed task plan with ordered steps. "
        "Provide the task plan in a structured JSON format."
    )
    return prompt

def construct_response_generation_prompt(reasoning_outputs: dict[str, Any]) -> str:
    """Construct a prompt for the Response Generation Agent.

    Args:
        reasoning_outputs (Dict[str, Any]): Outputs from the reasoning agent.

    Returns:
        str: The constructed prompt string.
    """
    prompt = (
        "You are a response generation agent.\n"
        f"Reasoning Outputs: {reasoning_outputs}\n"
        "Synthesize the outputs into a coherent and user-friendly response. "
        "Ensure the response addresses the user's original query and follows any provided guidelines."
    )
    return prompt

def construct_knowledge_integration_prompt(new_relations: dict[str, Any]) -> str:
    """Construct a prompt for the Dynamic Knowledge Integration Agent.

    Args:
        new_relations (Dict[str, Any]): New relations to be added to the knowledge graph.

    Returns:
        str: The constructed prompt string.
    """
    prompt = (
        "You are a dynamic knowledge integration agent.\n"
        f"New Relations: {new_relations}\n"
        "Update the knowledge graph with these relations, ensuring consistency and avoiding duplication. "
        "Provide confirmation once the integration is complete."
    )
    return prompt

def construct_extrapolation_prompt(entity1: str, relation: str, entity2: str, known_facts: list[str]) -> str:
    """Construct a prompt for the LLM to extrapolate the veracity of the relation.

    Args:
        entity1 (str): The first entity.
        relation (str): The relation to assess.
        entity2 (str): The second entity.
        known_facts (List[str]): A list of known facts related to the entities.

    Returns:
        str: The constructed prompt string.
    """
    facts_str = "\n".join(f"- {fact}" for fact in known_facts)
    prompt = (
        "You are a reasoning agent with advanced frameworks.\n"
        "Based on the following known facts:\n"
        f"{facts_str}\n"
        f"\nAssess the likelihood that '{entity1}' and '{entity2}' have the relation '{relation}'. "
        "Provide a detailed explanation using Chain-of-Thought reasoning and estimate the confidence level."
    )
    return prompt
