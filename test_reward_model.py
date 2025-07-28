import re

# Copy the reward model functions from datatest.py
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

# Test scenarios
print("="*80)
print("REWARD MODEL TEST SCENARIOS")
print("="*80)

# Expected answer from dataset
expected_answer = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"

test_cases = [
    {
        "name": "Perfect Response (Correct + Boxed)",
        "response": "The answer is $\\boxed{72}$ clips."
    },
    {
        "name": "Correct but No Boxed Format",
        "response": "The total number of clips is 72."
    },
    {
        "name": "Wrong Answer with Boxed Format",
        "response": "The answer is $\\boxed{96}$ clips."
    },
    {
        "name": "Wrong Answer without Boxed Format",
        "response": "The total is 100 clips."
    },
    {
        "name": "Complex Boxed Format",
        "response": "After calculation, we get \\boxed{48 + 24 = 72} clips total."
    },
    {
        "name": "No Clear Answer",
        "response": "I need to calculate the total clips sold."
    }
]

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"Response: {test['response']}")
    
    total_reward, correctness_reward, format_reward, details = calculate_reward(
        test['response'], 
        expected_answer
    )
    
    print(f"Total Reward: {total_reward:.2f}")
    print(f"  - Correctness: {correctness_reward:.2f}")
    print(f"  - Format: {format_reward:.2f}")
    print(f"Details: {details}")
    print("-" * 60) 