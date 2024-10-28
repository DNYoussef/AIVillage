"""Standardized formats for input/output handling."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OutputFormat:
    """Standard format for agent outputs."""
    content: str
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime = datetime.now()
    source: Optional[str] = None
    references: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    version: str = "1.0"

def create_standardized_output(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
    source: Optional[str] = None,
    references: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> OutputFormat:
    """
    Create standardized output format.
    
    Args:
        content: Main output content
        metadata: Additional metadata
        confidence: Confidence score (0-1)
        source: Source of the output
        references: Reference sources
        tags: Categorization tags
        
    Returns:
        Standardized output format
    """
    return OutputFormat(
        content=content,
        metadata=metadata or {},
        confidence=confidence,
        source=source,
        references=references,
        tags=tags
    )

def create_standardized_prompt(
    instruction: str,
    context: Optional[Union[str, Dict[str, Any]]] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[List[str]] = None,
    format_instructions: Optional[str] = None
) -> str:
    """
    Create standardized prompt format.
    
    Args:
        instruction: Main instruction/task
        context: Additional context
        examples: Example inputs/outputs
        constraints: Task constraints
        format_instructions: Output format instructions
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Add instruction
    prompt_parts.append(f"Instruction:\n{instruction}\n")
    
    # Add context if provided
    if context:
        if isinstance(context, str):
            prompt_parts.append(f"Context:\n{context}\n")
        else:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            prompt_parts.append(f"Context:\n{context_str}\n")
    
    # Add examples if provided
    if examples:
        prompt_parts.append("Examples:")
        for i, example in enumerate(examples, 1):
            example_str = "\n".join(f"{k}: {v}" for k, v in example.items())
            prompt_parts.append(f"Example {i}:\n{example_str}\n")
    
    # Add constraints if provided
    if constraints:
        prompt_parts.append("Constraints:")
        for constraint in constraints:
            prompt_parts.append(f"- {constraint}")
        prompt_parts.append("")
    
    # Add format instructions if provided
    if format_instructions:
        prompt_parts.append(f"Format Instructions:\n{format_instructions}\n")
    
    return "\n".join(prompt_parts)

def validate_output_format(output: OutputFormat) -> bool:
    """
    Validate output format.
    
    Args:
        output: Output format to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        if not output.content:
            return False
        
        # Check confidence range
        if not 0 <= output.confidence <= 1:
            return False
        
        # Check metadata is dict
        if not isinstance(output.metadata, dict):
            return False
        
        # Check optional lists
        if output.references and not isinstance(output.references, list):
            return False
        if output.tags and not isinstance(output.tags, list):
            return False
        
        # Check timestamp
        if not isinstance(output.timestamp, datetime):
            return False
        
        return True
    except Exception:
        return False

def merge_outputs(outputs: List[OutputFormat]) -> OutputFormat:
    """
    Merge multiple outputs into one.
    
    Args:
        outputs: List of outputs to merge
        
    Returns:
        Merged output format
    """
    if not outputs:
        raise ValueError("No outputs to merge")
    
    # Merge content
    merged_content = "\n".join(out.content for out in outputs)
    
    # Merge metadata
    merged_metadata = {}
    for out in outputs:
        merged_metadata.update(out.metadata)
    
    # Calculate average confidence
    avg_confidence = sum(out.confidence for out in outputs) / len(outputs)
    
    # Combine references and tags
    merged_refs = []
    merged_tags = []
    for out in outputs:
        if out.references:
            merged_refs.extend(out.references)
        if out.tags:
            merged_tags.extend(out.tags)
    
    # Remove duplicates
    merged_refs = list(dict.fromkeys(merged_refs))
    merged_tags = list(dict.fromkeys(merged_tags))
    
    return OutputFormat(
        content=merged_content,
        metadata=merged_metadata,
        confidence=avg_confidence,
        references=merged_refs or None,
        tags=merged_tags or None
    )

def format_as_json(output: OutputFormat) -> Dict[str, Any]:
    """
    Convert output format to JSON-compatible dict.
    
    Args:
        output: Output format to convert
        
    Returns:
        JSON-compatible dictionary
    """
    return {
        "content": output.content,
        "metadata": output.metadata,
        "confidence": output.confidence,
        "timestamp": output.timestamp.isoformat(),
        "source": output.source,
        "references": output.references,
        "tags": output.tags,
        "version": output.version
    }

def format_as_markdown(output: OutputFormat) -> str:
    """
    Convert output format to markdown string.
    
    Args:
        output: Output format to convert
        
    Returns:
        Markdown formatted string
    """
    md_parts = []
    
    # Add content
    md_parts.append(output.content)
    md_parts.append("")
    
    # Add metadata section
    if output.metadata:
        md_parts.append("## Metadata")
        for key, value in output.metadata.items():
            md_parts.append(f"- **{key}**: {value}")
        md_parts.append("")
    
    # Add references section
    if output.references:
        md_parts.append("## References")
        for ref in output.references:
            md_parts.append(f"- {ref}")
        md_parts.append("")
    
    # Add tags section
    if output.tags:
        md_parts.append("## Tags")
        md_parts.append(", ".join(f"`{tag}`" for tag in output.tags))
        md_parts.append("")
    
    # Add footer
    md_parts.append("---")
    md_parts.append(f"Confidence: {output.confidence:.2f}")
    if output.source:
        md_parts.append(f"Source: {output.source}")
    md_parts.append(f"Generated: {output.timestamp.isoformat()}")
    md_parts.append(f"Version: {output.version}")
    
    return "\n".join(md_parts)
