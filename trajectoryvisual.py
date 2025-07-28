from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import networkx as nx
from matplotlib.lines import Line2D

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
    enable_thinking=True
)

# Tokenize the input
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Token-by-token generation with alternatives tracking
def generate_with_alternatives(model, input_ids, max_new_tokens=30, temperature=0.6, top_p=0.95, top_k=20, num_alternatives=15):
    """Generate tokens one by one and track top alternatives at each step"""
    generation_tree = []
    current_ids = input_ids
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Get model outputs
            outputs = model(current_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for the last position
            
            # Apply temperature
            logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top alternatives (more than we'll sample from)
            top_alternatives_probs, top_alternatives_indices = torch.topk(probs, k=min(num_alternatives, probs.size(-1)))
            
            # Apply top-k filtering for actual sampling
            top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
            
            # Apply top-p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff for top-p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Get valid tokens after filtering
            valid_mask = ~sorted_indices_to_remove
            valid_probs = sorted_probs[valid_mask]
            valid_indices = sorted_indices[valid_mask]
            
            # Sample from the filtered distribution
            if len(valid_probs) > 0:
                sampled_idx = torch.multinomial(valid_probs, 1)
                chosen_token_idx = top_k_indices[valid_indices[sampled_idx]]
            else:
                # Fallback to most likely token
                chosen_token_idx = top_k_indices[0]
            
            # Store step info with alternatives
            step_info = {
                'step': step,
                'chosen_token_id': chosen_token_idx.item(),
                'chosen_token_text': tokenizer.decode([chosen_token_idx.item()], skip_special_tokens=False),
                'chosen_token_prob': probs[chosen_token_idx].item(),
                'alternatives': []
            }
            
            # Store all top alternatives
            for i in range(len(top_alternatives_indices)):
                alt_token_id = top_alternatives_indices[i].item()
                alt_token_text = tokenizer.decode([alt_token_id], skip_special_tokens=False)
                alt_token_prob = top_alternatives_probs[i].item()
                is_chosen = (alt_token_id == chosen_token_idx.item())
                
                step_info['alternatives'].append({
                    'token_id': alt_token_id,
                    'token_text': alt_token_text,
                    'probability': alt_token_prob,
                    'is_chosen': is_chosen
                })
            
            generation_tree.append(step_info)
            
            # Update current_ids for next iteration
            next_token = chosen_token_idx.view(1, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for EOS token
            if chosen_token_idx.item() == tokenizer.eos_token_id:
                break
    
    return generation_tree

# Create branching tree visualization
def create_branching_tree_visualization(generation_tree, filename, title_suffix="", max_alternatives_shown=10):
    """Create a branching tree visualization of token generation"""
    fig, ax = plt.subplots(1, 1, figsize=(30, 12))  # Wider for 30 tokens
    
    # Layout parameters
    x_spacing = 2.5
    y_spacing = 0.8
    box_width = 2.2
    box_height = 0.6
    
    # Colors
    chosen_color = '#2ecc71'  # Green for chosen path
    alternative_color = '#ecf0f1'  # Light gray for alternatives
    high_prob_color = '#3498db'  # Blue for high probability alternatives
    
    # Draw the tree
    for step_idx, step_info in enumerate(generation_tree):
        x_pos = step_idx * x_spacing
        
        # Draw alternatives
        alternatives = step_info['alternatives'][:max_alternatives_shown]
        
        for alt_idx, alt in enumerate(alternatives):
            y_pos = alt_idx * y_spacing - (len(alternatives) - 1) * y_spacing / 2
            
            # Determine color based on probability and whether it's chosen
            if alt['is_chosen']:
                color = chosen_color
                edge_width = 3
                edge_color = 'darkgreen'
            elif alt['probability'] > 0.1:
                color = high_prob_color
                edge_width = 1
                edge_color = 'darkblue'
            else:
                color = alternative_color
                edge_width = 1
                edge_color = 'gray'
            
            # Create box for token
            box = FancyBboxPatch(
                (x_pos - box_width/2, y_pos - box_height/2), 
                box_width, 
                box_height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width
            )
            ax.add_patch(box)
            
            # Add token text
            token_text = alt['token_text'].replace('\n', '\\n')[:15]  # Truncate long tokens
            ax.text(x_pos, y_pos + 0.1, token_text, 
                   ha='center', va='center', fontsize=8, weight='bold' if alt['is_chosen'] else 'normal')
            
            # Add probability
            ax.text(x_pos, y_pos - 0.2, f'{alt["probability"]:.3f}', 
                   ha='center', va='center', fontsize=6, style='italic')
            
            # Draw connection to next chosen token
            if alt['is_chosen'] and step_idx < len(generation_tree) - 1:
                next_chosen_idx = None
                next_alternatives = generation_tree[step_idx + 1]['alternatives'][:max_alternatives_shown]
                for next_idx, next_alt in enumerate(next_alternatives):
                    if next_alt['is_chosen']:
                        next_chosen_idx = next_idx
                        break
                
                if next_chosen_idx is not None:
                    next_x = (step_idx + 1) * x_spacing
                    next_y = next_chosen_idx * y_spacing - (len(next_alternatives) - 1) * y_spacing / 2
                    
                    # Draw arrow
                    ax.annotate('', xy=(next_x - box_width/2, next_y), 
                               xytext=(x_pos + box_width/2, y_pos),
                               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    
    # Set title and labels
    ax.set_title(f'Token Generation Branching Tree{title_suffix}\n(Top 10 alternatives shown per step)', 
                fontsize=16, weight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Chosen Token',
               markerfacecolor=chosen_color, markersize=10, markeredgecolor='darkgreen', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', label='High Prob Alternative (>0.1)',
               markerfacecolor=high_prob_color, markersize=10, markeredgecolor='darkblue'),
        Line2D([0], [0], marker='o', color='w', label='Low Prob Alternative',
               markerfacecolor=alternative_color, markersize=10, markeredgecolor='gray')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Adjust axes
    ax.set_xlim(-1, len(generation_tree) * x_spacing)
    ax.set_ylim(-max_alternatives_shown * y_spacing / 2 - 1, max_alternatives_shown * y_spacing / 2 + 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid showing in notebook

# Function to run generation and visualization with given parameters
def run_generation_and_viz(temperature, top_p, top_k, run_number):
    print(f"\n{'='*80}")
    print(f"RUN {run_number}: Temperature={temperature}, TopP={top_p}, TopK={top_k}")
    print('='*80)
    
    # Generate with specified parameters
    generation_tree = generate_with_alternatives(
        model, 
        model_inputs.input_ids, 
        max_new_tokens=30,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_alternatives=15
    )
    
    # Create visualization
    filename = f'token_branching_tree_{run_number}.png'
    title_suffix = f" - Run {run_number} (T={temperature}, P={top_p}, K={top_k})"
    create_branching_tree_visualization(generation_tree, filename, title_suffix)
    
    # Print generated text
    generated_text = ''.join([step['chosen_token_text'] for step in generation_tree])
    print(f"\nGenerated text (30 tokens):")
    print(generated_text)
    
    # Statistics
    chosen_probs = [step['chosen_token_prob'] for step in generation_tree]
    print(f"\nStatistics:")
    print(f"- Average chosen token probability: {np.mean(chosen_probs):.3f}")
    print(f"- Min chosen token probability: {np.min(chosen_probs):.3f}")
    print(f"- Max chosen token probability: {np.max(chosen_probs):.3f}")
    print(f"- Saved visualization to: {filename}")
    
    return generation_tree

# Run 1: Standard parameters (as recommended for thinking mode)
print("\nGenerating two different sampling trajectories...")
tree1 = run_generation_and_viz(temperature=0.6, top_p=0.95, top_k=20, run_number=1)

# Run 2: Lower temperature for more deterministic generation
tree2 = run_generation_and_viz(temperature=0.3, top_p=0.9, top_k=10, run_number=2)

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
print(f"Two visualizations saved:")
print(f"- token_branching_tree_1.png (Temperature=0.6, more exploratory)")
print(f"- token_branching_tree_2.png (Temperature=0.3, more focused)")
print("\nThe lower temperature run should show more concentrated probability mass on fewer tokens,")
    
    
    

    
