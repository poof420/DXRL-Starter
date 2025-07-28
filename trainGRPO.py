

import os
import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb
from typing import List, Dict, Tuple, Optional
import re
import os



# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "./grpo_GSM8K_0.6B"
WANDB_PROJECT = "grpo_class_GSM8K"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOCAL_RANK"] = "0"


# ============================================================================
# Helper Functions and Setup
# ============================================================================

def extract_dataset_answer(answer_text):
    """Extract the numerical answer from GSM8K dataset format (after ####)"""
    match = re.search(r'####\s*(\d+)', answer_text)
    if match:
        return int(match.group(1))
    return None

def extract_model_answer(response_text):
    """Extract answer from model's response, looking for \boxed{} format"""
    # Look for \boxed{...} pattern
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response_text)
    if boxed_match:
        answer_text = boxed_match.group(1)
        # Try to extract just the number from the boxed content
        number_match = re.search(r'(\d+)', answer_text)
        if number_match:
            return int(number_match.group(1))
    
    # Fallback: look for any number that might be the final answer
    # (This is less reliable but catches cases without proper formatting)
    numbers = re.findall(r'\b(\d+)\b', response_text)
    if numbers:
        # Return the last number found as a guess
        return int(numbers[-1])
    
    return None

def calculate_reward_score(model_response, expected_answer_text):
    """
    Calculate reward for model response.
    - 0.9 points for correct answer
    - 0.1 points for using \boxed{} format
    Returns: total_reward
    """
    # Extract expected answer
    expected_answer = extract_dataset_answer(expected_answer_text)
    if expected_answer is None:
        return 0.0
    
    # Check if model used boxed format
    has_boxed_format = bool(re.search(r'\\boxed\{[^}]+\}', model_response))
    format_reward = 0.1 if has_boxed_format else 0.0
    
    # Extract model's answer
    model_answer = extract_model_answer(model_response)
    if model_answer is None:
        return format_reward  # Only format reward if no answer found
    
    # Check correctness
    is_correct = (model_answer == expected_answer)
    correctness_reward = 0.9 if is_correct else 0.0
    
    # Total reward
    total_reward = correctness_reward + format_reward
    
    return total_reward


# ============================================================================
# GRPO Reward Function
# ============================================================================

def gsm8k_reward_function(prompts, completions, answer, **kwargs):
    """
    Reward function for GSM8K problems.
    
    Args:
        prompts: List of prompts (questions)
        completions: List of model completions (answers)
        answer: List of expected answers from the dataset
        **kwargs: Additional arguments (including trainer_state if needed)
    
    Returns:
        List of reward scores (floats)
    """
    rewards = []
    
    # Handle case where answer might be repeated for multiple generations
    answers_list = answer
    if len(answer) != len(completions):
        # Repeat answers to match completions if needed
        answers_list = answer * (len(completions) // len(answer))
    
    for i in range(len(completions)):
        # Calculate reward for this completion
        reward = calculate_reward_score(completions[i], answers_list[i % len(answer)])
        rewards.append(reward)
    
    return rewards


# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_dataset(dataset, tokenizer):
    """
    Prepare the GSM8K dataset for GRPO training.
    Formats the questions as prompts with instruction.
    """
    formatted_examples = []
    for example in dataset:
        user_message = f"Please solve this math problem step by step: {example['question']}\n\nPlease put your final answer in \\boxed{{}} format."
        messages = [
            {"role": "user", "content": user_message}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode for complex math reasoning
        )
        formatted_examples.append({
            "prompt": prompt,
            "question": example['question'],
            "answer": example['answer']
        })
    formatted_dataset = formatted_examples
    return formatted_dataset


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    
    # Initialize wandb
    wandb.init(project=WANDB_PROJECT, name=f"grpo-{MODEL_NAME.split('/')[-1]}")
    
    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main")
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    print(f'Training set size: {len(train_dataset)}')
    print(f'Test set size: {len(eval_dataset)}\n')


    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, model_max_length=6144)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to left for generation
    tokenizer.padding_side = 'left'

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    
    # Show a sample
    sample = train_dataset[0]
    print("Sample from prepared dataset:")
    print(f"Prompt: {sample['prompt'][:200]}...")
    print(f"Answer: {sample['answer']}\n")
    print("-" * 80 + "\n")
    
    # Create GRPO configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        
        # Basic training parameters
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,  # Reduce from 4 to 1
        eval_accumulation_steps=8,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_train_epochs=1,
        warmup_steps=10,
        
        # Logging and saving
        logging_steps=1,
        save_steps=150,
        eval_steps=None,     # Change from 500 to None
        
        # Generation parameters
        num_generations=8,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_prompt_length=2048,
        max_completion_length=1024,  # Reduce from 4096
        
        # vLLM configuration
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.15,
        
        # GRPO specific parameters
        beta=0.0,  # No KL penalty as recommended in recent papers
        epsilon=0.2,  # Standard epsilon value
        epsilon_high=0.28,  # Higher epsilon for upper bound
        loss_type="dr_grpo",  # Use dr_grpo loss type to eliminate length bias
        
        # Optimization
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        use_liger_kernel=True,
        
        # Tracking
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=2,
        wandb_log_unique_prompts=True,
        run_name=f"grpo-{MODEL_NAME.split('/')[-1]}_gsm8k",
        mask_truncated_completions=True,
        repetition_penalty=1.1,  # Penalize repetition, may affect length
    )
    
    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=gsm8k_reward_function
    )
    
    # Start training
    print("\n🚀 Starting GRPO training...")
    print(f"Total training steps: {trainer.state.max_steps if hasattr(trainer.state, 'max_steps') else 'auto'}")
    print(f"Using vLLM in '{training_args.vllm_mode}' mode on {torch.cuda.device_count()} GPUs.")
    
    # Initial evaluation
    # print("\n📊 Running initial evaluation...")
    # eval_results = trainer.evaluate()
    # print("Initial evaluation results:", eval_results)
    
    # Train the model
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    
    print("\n✅ Training complete!")
    
    # Final evaluation
    print("\n📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    print("Final evaluation results:", eval_results)
    
    # Log final results to wandb
    wandb.log({"final_eval": eval_results})
    wandb.finish()
    
if __name__ == "__main__":
    main()