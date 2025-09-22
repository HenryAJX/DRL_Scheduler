import torch
import argparse
import os
from safetensors.torch import save_file

def convert_checkpoint(input_path: str, output_path: str):
    """
    Loads a PyTorch checkpoint (.pt) and saves it in the .safetensors format.

    This function flattens the nested state dictionaries (e.g., 'actor_state_dict')
    into a single-level dictionary suitable for the safetensors format.
    For example, a weight named 'layer1.weight' inside 'actor_state_dict'
    becomes 'actor_state_dict.layer1.weight'.
    """
    print(f"Loading PyTorch checkpoint from: {input_path}")
    try:
        # Load the checkpoint from the .pt file
        checkpoint = torch.load(input_path, map_location="cpu")
    except FileNotFoundError:
        # --- THIS LINE IS NOW FIXED ---
        print(f"Error: Input file not found at {input_path}") # Was input_t, now corrected to input_path
        return

    # Create a new, flattened dictionary to hold all tensors
    flat_tensors = {}

    # Iterate through the top-level keys of the checkpoint (e.g., 'actor_state_dict')
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            # This handles single tensors like 'log_alpha'
            flat_tensors[key] = value
        elif isinstance(value, dict):
            # This handles nested state_dicts
            for sub_key, tensor in value.items():
                # Create a new key by joining the parent and child keys
                flat_key = f"{key}.{sub_key}"
                flat_tensors[flat_key] = tensor
    
    print(f"Successfully loaded and flattened {len(flat_tensors)} tensors.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the flattened dictionary to a .safetensors file
    print(f"Saving to .safetensors format at: {output_path}")
    save_file(flat_tensors, output_path)
    print("✅ Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt checkpoints to .safetensors format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .pt checkpoint file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .safetensors file.")
    
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output)