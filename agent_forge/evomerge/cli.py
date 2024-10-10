import argparse
import json
import logging
import os
from .config import create_default_config, Configuration, ModelReference
from .evolutionary_tournament import run_evolutionary_tournament
from .utils import generate_text, evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from .. import AgentForge

def generate_config_interactive():
    config = {}
    config['merge_method'] = input("Enter merge method (ps/dfs/ps_dfs): ")
    config['num_generations'] = int(input("Enter number of generations: "))
    config['population_size'] = int(input("Enter population size: "))
    config['mutation_rate'] = float(input("Enter mutation rate (0-1): "))
    config['tournament_size'] = int(input("Enter tournament size: "))
    config['early_stopping_generations'] = int(input("Enter early stopping generations: "))
    return config

def is_local_path(path):
    return os.path.isdir(path)

def main():
    parser = argparse.ArgumentParser(description="EvoMerge: Evolutionary Model Merging System")
    parser.add_argument("--config", type=str, help="Path to a JSON configuration file")
    parser.add_argument("--run", action="store_true", help="Run the evolutionary tournament")
    parser.add_argument("--evaluate", type=str, help="Evaluate a merged model at the given path")
    parser.add_argument("--generate", type=str, help="Generate text using a merged model at the given path")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt for text generation")
    parser.add_argument("--model1", type=str, help="Local path or Hugging Face model ID for the first model")
    parser.add_argument("--model2", type=str, help="Local path or Hugging Face model ID for the second model")
    parser.add_argument("--model3", type=str, help="Local path or Hugging Face model ID for the third model")
    parser.add_argument("--generate-config", action="store_true", help="Generate a configuration file interactively")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    if args.generate_config:
        config = generate_config_interactive()
        with open('evomerge_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Configuration file generated: evomerge_config.json")
        return

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Configuration(**config_dict)
    else:
        config = create_default_config()

    # Update the configuration with the provided models
    if args.model1 or args.model2 or args.model3:
        new_models = []
        for i, model_path in enumerate([args.model1, args.model2, args.model3], start=1):
            if model_path:
                model_path = model_path if is_local_path(model_path) else f"Qwen/{model_path}"
                new_models.append(ModelReference(name=f"model{i}", path=model_path))
        
        if len(new_models) < 2:
            logger.error("Error: At least two models must be provided for merging.")
            return
        
        config.models = new_models

    if args.run:
        logger.info("Running evolutionary tournament...")
        agent_forge = AgentForge(model_name=config.models[0].path)  # Use the first model for RAGPromptBaker
        best_model_path = agent_forge.run_full_agent_forge_process()
        logger.info(f"Best model saved at: {best_model_path}")
        
        logger.info("\nEvaluating best model:")
        evaluation_result = evaluate_model(best_model_path)
        logger.info(f"Overall score: {evaluation_result['overall_score']:.2f}")
        logger.info("\nDetailed results:")
        for task, result in evaluation_result['results'].items():
            logger.info(f"\n{task} Task:")
            logger.info(result)

        logger.info("\nGenerating sample text:")
        model = AutoModelForCausalLM.from_pretrained(best_model_path)
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        generated_text = generate_text(model, tokenizer, args.prompt)
        logger.info(f"Generated text: {generated_text}")

    elif args.evaluate:
        logger.info(f"Evaluating model at {args.evaluate}")
        evaluation_result = evaluate_model(args.evaluate)
        logger.info(f"Overall score: {evaluation_result['overall_score']:.2f}")
        logger.info("\nDetailed results:")
        for task, result in evaluation_result['results'].items():
            logger.info(f"\n{task} Task:")
            logger.info(result)

    elif args.generate:
        logger.info(f"Generating text using model at {args.generate}")
        model = AutoModelForCausalLM.from_pretrained(args.generate)
        tokenizer = AutoTokenizer.from_pretrained(args.generate)
        generated_text = generate_text(model, tokenizer, args.prompt)
        logger.info(f"Generated text: {generated_text}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
