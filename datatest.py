from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Reward Model Functions
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

def calculate_reward(model_response, expected_answer_text):
    """
    Calculate reward for model response.
    - 0.9 points for correct answer
    - 0.1 points for using \boxed{} format
    Returns: (total_reward, correctness_reward, format_reward, details)
    """
    # Extract expected answer
    expected_answer = extract_dataset_answer(expected_answer_text)
    if expected_answer is None:
        return 0.0, 0.0, 0.0, "Could not extract expected answer from dataset"
    
    # Check if model used boxed format
    has_boxed_format = bool(re.search(r'\\boxed\{[^}]+\}', model_response))
    format_reward = 0.1 if has_boxed_format else 0.0
    
    # Extract model's answer
    model_answer = extract_model_answer(model_response)
    if model_answer is None:
        return format_reward, 0.0, format_reward, f"No answer found in model response (format reward: {format_reward})"
    
    # Check correctness
    is_correct = (model_answer == expected_answer)
    correctness_reward = 0.9 if is_correct else 0.0
    
    # Total reward
    total_reward = correctness_reward + format_reward
    
    # Create detailed message
    details = f"Model answer: {model_answer}, Expected: {expected_answer}, "
    details += f"Correct: {is_correct}, Used \\boxed format: {has_boxed_format}"
    
    return total_reward, correctness_reward, format_reward, details

# Load GSM8K dataset
print("Loading GSM8K dataset...")
ds = load_dataset("openai/gsm8k", "main")
train, test = ds["train"], ds["test"]

print(f'Training set size: {len(train)}')
print(f'Test set size: {len(test)}\n')

# Get a sample problem from the dataset
sample_idx = 0  # First sample
sample_question = train[sample_idx]["question"]
sample_answer = train[sample_idx]["answer"]

print("Sample GSM8K Problem:")
print(f"Question: {sample_question}")
print(f"Expected Answer: {sample_answer}\n")
print("-" * 80 + "\n")

# Load Qwen3-0.6B model
model_name = "Qwen/Qwen3-0.6B"
print(f"Loading {model_name} model and tokenizer...")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Prepare the prompt with the GSM8K question
prompt = f"Please solve this math problem step by step: {sample_question}"
messages = [
    {"role": "user", "content": prompt}
]

# Apply chat template with thinking mode enabled
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Enable thinking mode for complex math reasoning
)

# Tokenize the input
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating response with Qwen3-0.6B in thinking mode...")
print("(Using recommended parameters: Temperature=0.6, TopP=0.95, TopK=20, MinP=0)")
# What the generation settings mean:
# - temperature: How "random" or creative the model's answers are. Lower numbers (like 0.2) make it stick closely to the most likely answer. Higher numbers (like 1.0) make it more likely to try unusual or creative responses.
# - top_p: The model picks from the smallest group of words that together make up a certain percentage (like 95%) of the total probability. This helps keep answers sensible but still allows some variety.
# - top_k: The model only considers the top "k" most likely next words (for example, the top 20) when choosing what to say next. This limits its choices to the most likely options.
# - min_p: The model ignores any words that are extremely unlikely (below a certain probability, like 0). This helps avoid strange or irrelevant words.


# Generate with recommended parameters for thinking mode
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    do_sample=True,  # Must use sampling, not greedy decoding
)

# Extract output tokens
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# Parse thinking content and regular content
try:
    # Find the </think> token (ID: 151668)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    # If </think> token not found, set index to 0
    index = 0

# Decode the thinking and content parts separately
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# Calculate reward
total_reward, correctness_reward, format_reward, details = calculate_reward(content, sample_answer)

print("\n" + "=" * 80)
print("THINKING CONTENT (internal reasoning):")
print("=" * 80)
print(thinking_content)
print("\n" + "=" * 80)
print("FINAL ANSWER:")
print("=" * 80)
print(content)
print("\n" + "=" * 80)
print("EXPECTED ANSWER FROM DATASET:")
print("=" * 80)
print(sample_answer)
print("\n" + "=" * 80)
print("REWARD MODEL EVALUATION:")
print("=" * 80)
print(f"Total Reward: {total_reward:.2f} / 1.00")
print(f"  - Correctness Reward: {correctness_reward:.2f} / 0.90")
print(f"  - Format Reward: {format_reward:.2f} / 0.10")
print(f"Details: {details}")
    
    
    

    
