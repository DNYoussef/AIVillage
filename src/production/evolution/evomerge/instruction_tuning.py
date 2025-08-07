import logging

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
)

from .config import MergeSettings
from .merging.merge_techniques import MERGE_TECHNIQUES
from .utils import EvoMergeException

logger = logging.getLogger(__name__)


def is_instruction_tuned_model(
    model: torch.nn.Module, tokenizer: AutoTokenizer
) -> bool:
    instruction_keywords = [
        "instruction",
        "task",
        "input",
        "output",
        "human",
        "assistant",
    ]
    special_tokens = tokenizer.all_special_tokens
    vocab = tokenizer.get_vocab()

    keyword_presence = sum(1 for keyword in instruction_keywords if keyword in vocab)
    special_token_presence = sum(
        1
        for token in special_tokens
        if any(keyword in token.lower() for keyword in instruction_keywords)
    )

    return (keyword_presence >= 3) or (special_token_presence >= 2)


def merge_instruction_tuned_models(
    models: list[torch.nn.Module],
    tokenizers: list[AutoTokenizer],
    merge_settings: MergeSettings,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    try:
        logger.info("Starting instruction-tuned model merge")

        if merge_settings.merge_method == "ps":
            merged_model = merge_ps(models, merge_settings)
        elif merge_settings.merge_method == "dfs":
            merged_model = merge_dfs(models, merge_settings)
        elif merge_settings.merge_method == "ps_dfs":
            merged_model = merge_ps_dfs(models, merge_settings)
        else:
            msg = f"Unsupported merge method for instruction-tuned models: {merge_settings.merge_method}"
            raise EvoMergeException(msg)

        # Merge tokenizers
        merged_tokenizer = merge_tokenizers(tokenizers)

        # Preserve instruction-tuning capabilities
        merged_model = preserve_instruction_tuning(merged_model, models, tokenizers)

        return merged_model, merged_tokenizer
    except Exception as e:
        logger.exception(f"Failed to merge instruction-tuned models: {e!s}")
        msg = f"Failed to merge instruction-tuned models: {e!s}"
        raise EvoMergeException(msg)


def merge_ps(
    models: list[torch.nn.Module], merge_settings: MergeSettings
) -> torch.nn.Module:
    logger.info("Performing parameter space merge")
    merged_state_dict = {}
    for technique in merge_settings.ps_techniques:
        if technique not in MERGE_TECHNIQUES:
            msg = f"Unknown merge technique: {technique}"
            raise ValueError(msg)

        technique_params = merge_settings.parameters.get(technique, {})
        weights = technique_params.pop("weights", None)

        for name, param in models[0].named_parameters():
            if name not in merged_state_dict:
                merged_state_dict[name] = torch.zeros_like(param.data)

            [model.state_dict()[name] for model in models]
            if weights is not None:
                merged_state_dict = MERGE_TECHNIQUES[technique](
                    merged_state_dict, models, weights=weights, **technique_params
                )

    config = AutoConfig.from_pretrained(models[0].config._name_or_path)
    merged_model = AutoModelForCausalLM.from_config(config)
    merged_model.load_state_dict(merged_state_dict)
    return merged_model


def merge_dfs(
    models: list[torch.nn.Module], merge_settings: MergeSettings
) -> torch.nn.Module:
    logger.info("Performing deep fusion space merge")
    config = AutoConfig.from_pretrained(models[0].config._name_or_path)
    merged_model = AutoModelForCausalLM.from_config(config)

    for technique in merge_settings.dfs_techniques:
        if technique not in MERGE_TECHNIQUES:
            msg = f"Unknown merge technique: {technique}"
            raise ValueError(msg)
        merged_model = MERGE_TECHNIQUES[technique](
            merged_model, models, **merge_settings.parameters.get(technique, {})
        )
    return merged_model


def merge_ps_dfs(
    models: list[torch.nn.Module], merge_settings: MergeSettings
) -> torch.nn.Module:
    logger.info("Performing combined parameter space and deep fusion space merge")
    ps_model = merge_ps(models, merge_settings)
    return merge_dfs([ps_model, *models], merge_settings)


def merge_tokenizers(tokenizers: list[AutoTokenizer]) -> AutoTokenizer:
    logger.info("Merging tokenizers")
    base_tokenizer = tokenizers[0]
    for tokenizer in tokenizers[1:]:
        base_tokenizer.add_tokens(
            list(
                set(tokenizer.get_vocab().keys())
                - set(base_tokenizer.get_vocab().keys())
            )
        )
    return base_tokenizer


def preserve_instruction_tuning(
    merged_model: torch.nn.Module,
    original_models: list[torch.nn.Module],
    tokenizers: list[AutoTokenizer],
) -> torch.nn.Module:
    logger.info("Preserving instruction-tuning capabilities")
    instruction_embeddings = []
    for model, tokenizer in zip(original_models, tokenizers, strict=False):
        instruction_tokens = [
            token
            for token in tokenizer.get_vocab()
            if "instruction" in token.lower() or "task" in token.lower()
        ]
        instruction_ids = tokenizer.convert_tokens_to_ids(instruction_tokens)
        instruction_embeddings.append(
            model.get_input_embeddings().weight[instruction_ids]
        )

    avg_instruction_embedding = torch.mean(torch.stack(instruction_embeddings), dim=0)

    merged_tokenizer = merge_tokenizers(tokenizers)
    merged_instruction_tokens = [
        token
        for token in merged_tokenizer.get_vocab()
        if "instruction" in token.lower() or "task" in token.lower()
    ]
    merged_instruction_ids = merged_tokenizer.convert_tokens_to_ids(
        merged_instruction_tokens
    )

    with torch.no_grad():
        merged_model.get_input_embeddings().weight[
            merged_instruction_ids
        ] = avg_instruction_embedding

    return merged_model


def fine_tune_on_instructions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    instruction_dataset: list[str],
    num_epochs: int = 3,
):
    """Fine-tune the merged model on instruction-following tasks."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Prepare the dataset
    encoded_dataset = [
        tokenizer.encode(text, truncation=True, max_length=512)
        for text in instruction_dataset
    ]
    dataset = TextDataset(encoded_dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch["input_ids"].to(model.device)
            labels = inputs.clone()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    return model


def evaluate_instruction_following(
    model: torch.nn.Module, tokenizer: AutoTokenizer, test_instructions: list[str]
) -> float:
    """Evaluate the instruction-following capabilities of the merged model."""
    model.eval()
    correct_responses = 0

    for instruction in test_instructions:
        inputs = tokenizer.encode(instruction, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Simple heuristic: check if the response contains key instruction-related words
        if any(
            keyword in response.lower()
            for keyword in ["answer", "result", "output", "response"]
        ):
            correct_responses += 1

    accuracy = correct_responses / len(test_instructions)
    return accuracy


# Example usage
def merge_and_evaluate_instruction_models(
    models: list[torch.nn.Module],
    tokenizers: list[AutoTokenizer],
    merge_settings: MergeSettings,
    instruction_dataset: list[str],
    test_instructions: list[str],
) -> tuple[torch.nn.Module, float]:
    merged_model, merged_tokenizer = merge_instruction_tuned_models(
        models, tokenizers, merge_settings
    )
    fine_tuned_model = fine_tune_on_instructions(
        merged_model, merged_tokenizer, instruction_dataset
    )
    accuracy = evaluate_instruction_following(
        fine_tuned_model, merged_tokenizer, test_instructions
    )
    print(f"Instruction-following accuracy: {accuracy:.2f}")
    return fine_tuned_model, accuracy
