#!/usr/bin/env python
"""Test script to verify the GRPO reward function works correctly."""

from trainGRPO import gsm8k_reward_function, prepare_dataset
from datasets import load_dataset

# Test the reward function with various examples
test_prompts = [
    "Please solve this math problem step by step: What is 2+2?",
    "Please solve this math problem step by step: What is 3*4?",
    "Please solve this math problem step by step: What is 10-5?",
]

test_completions = [
    "Let me solve this step by step. 2+2 = 4. Therefore, the answer is \\boxed{4}.",  # Correct with boxed
    "3 times 4 equals 12.",  # Correct but no boxed
    "10 minus 5 is \\boxed{6}.",  # Wrong with boxed
]

test_answers = [
    "2+2 = <<2+2=4>>4\n#### 4",
    "3*4 = <<3*4=12>>12\n#### 12", 
    "10-5 = <<10-5=5>>5\n#### 5",
]

print("Testing GRPO reward function...")
print("="*60)

rewards = gsm8k_reward_function(
    prompts=test_prompts,
    completions=test_completions,
    answer=test_answers
)

for i, (prompt, completion, answer, reward) in enumerate(zip(test_prompts, test_completions, test_answers, rewards)):
    print(f"\nTest {i+1}:")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Completion: {completion}")
    print(f"Expected answer: {answer.split('####')[1].strip()}")
    print(f"Reward: {reward}")
    print(f"Breakdown: Correct={'4' in completion and i==0 or '12' in completion and i==1 or '5' in completion and i==2}, "
          f"Boxed={'\\boxed' in completion}")

print("\n" + "="*60)
print("Expected rewards:")
print("Test 1: 1.0 (correct + boxed)")
print("Test 2: 0.9 (correct, no boxed)")
print("Test 3: 0.1 (wrong + boxed)")

# Test dataset preparation
print("\n" + "="*60)
print("Testing dataset preparation...")
ds = load_dataset("openai/gsm8k", "main", split="train[:3]")
prepared = prepare_dataset(ds)

for i in range(3):
    print(f"\nSample {i+1}:")
    print(f"Original question: {ds[i]['question'][:100]}...")
    print(f"Prepared prompt: {prepared[i]['prompt'][:150]}...")
    print(f"Answer preserved: {prepared[i]['answer'][:50]}...")

print("\n✅ Test complete!") 