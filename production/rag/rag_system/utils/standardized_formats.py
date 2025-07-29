from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    LIST = "list"
    DICT = "dict"

class StandardizedPrompt(BaseModel):
    task: str = Field(..., description="The main task or question to be addressed")
    context: str = Field(..., description="Any relevant context or background information")
    constraints: list[str] = Field(default=[], description="Any constraints or limitations to consider")
    examples: list[dict[str, str]] = Field(default=[], description="Optional examples to guide the response")
    output_format: OutputFormat = Field(..., description="The desired format for the output")
    additional_instructions: str = Field(default="", description="Any additional instructions or guidelines")
    metadata: dict[str, Any] = Field(default={}, description="Any additional metadata for the prompt")

    def to_string(self) -> str:
        prompt = f"Task: {self.task}\n\n"
        prompt += f"Context: {self.context}\n\n"

        if self.constraints:
            prompt += "Constraints:\n"
            for i, constraint in enumerate(self.constraints, 1):
                prompt += f"{i}. {constraint}\n"
            prompt += "\n"

        if self.examples:
            prompt += "Examples:\n"
            for i, example in enumerate(self.examples, 1):
                prompt += f"Example {i}:\n"
                for key, value in example.items():
                    prompt += f"  {key}: {value}\n"
                prompt += "\n"

        prompt += f"Output Format: {self.output_format.value}\n\n"

        if self.additional_instructions:
            prompt += f"Additional Instructions: {self.additional_instructions}\n\n"

        prompt += "Please provide your response based on the above information and guidelines."

        return prompt

class StandardizedOutput(BaseModel):
    task: str = Field(..., description="The original task or question addressed")
    response: str | list[str] | dict[str, Any] = Field(..., description="The main response content")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the response (0-1)")
    sources: list[str] = Field(default=[], description="Sources or references used in the response")
    metadata: dict[str, Any] = Field(default={}, description="Any additional metadata about the response")
    reasoning: str = Field(default="", description="Explanation of the reasoning process")
    uncertainty: float = Field(default=0.0, ge=0, le=1, description="Uncertainty level of the response (0-1)")
    alternative_responses: list[dict[str, Any]] = Field(default=[], description="Alternative responses or perspectives")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "response": self.response,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
            "reasoning": self.reasoning,
            "uncertainty": self.uncertainty,
            "alternative_responses": self.alternative_responses
        }

def create_standardized_prompt(
    task: str,
    context: str,
    output_format: OutputFormat,
    constraints: list[str] = [],
    examples: list[dict[str, str]] = [],
    additional_instructions: str = "",
    metadata: dict[str, Any] = {}
) -> StandardizedPrompt:
    return StandardizedPrompt(
        task=task,
        context=context,
        constraints=constraints,
        examples=examples,
        output_format=output_format,
        additional_instructions=additional_instructions,
        metadata=metadata
    )

def create_standardized_output(
    task: str,
    response: str | list[str] | dict[str, Any],
    confidence: float,
    sources: list[str] = [],
    metadata: dict[str, Any] = {},
    reasoning: str = "",
    uncertainty: float = 0.0,
    alternative_responses: list[dict[str, Any]] = []
) -> StandardizedOutput:
    return StandardizedOutput(
        task=task,
        response=response,
        confidence=confidence,
        sources=sources,
        metadata=metadata,
        reasoning=reasoning,
        uncertainty=uncertainty,
        alternative_responses=alternative_responses
    )
