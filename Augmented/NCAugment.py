# Plain and single-threaded vanilla NCA. No parallelisation. Data augmentation similar to Hierarchichal Reasoning Model.
import collections
import os
import json
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import sys
import subprocess

import datetime
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S") # Format: YYMMDD_HHMMSS

# --- 1. SETTINGS & PATHS ---
# Uncomment for local mac silicon run
ARC_DATA_DIR = "../dataset/script-tests/grouped-tasks-00576224"
OUTPUT_DIR = os.path.join("../runs", f"test_{timestamp}")
VISUALISE = True # Set to True to generate visualization.pdf at the end of execution. Only for single-threaded NCA.
EVALUATE_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../nca-code/", "evaluate.py"))

# Uncomment for Kaggle
# ARC_DATA_DIR = "/kaggle/input/arc-prize-2024"
# OUTPUT_DIR = "/kaggle/working"
# VISUALISE = True   # Set to True to generate visualization.pdf at the end of execution. Only for single-threaded NCA.
# EVALUATE_SCRIPT_PATH = "/kaggle/working/evaluate.py"

# Uncomment for Google Colab run
# from google.colab import drive
# drive.mount('/content/drive')
# ARC_DATA_DIR = "/content/drive/MyDrive/Cracking-ARC-AGI/dataset/script-tests/grouped-tasks"
# OUTPUT_DIR = os.path.join("/content/drive/MyDrive/Cracking-ARC-AGI/NCAs", f"test_{timestamp}")
# VISUALISE = True   # Set to True to generate visualization.pdf at the end of execution. Only for single-threaded NCA.
# EVALUATE_SCRIPT_PATH = "/content/drive/MyDrive/Cracking-ARC-AGI/evaluate.py"

INPUT_JSON_FILE = os.path.join(ARC_DATA_DIR, "challenges.json")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---Augmentation Flags & Settings ---
ENABLE_STATIC_AUGMENTATION = True     # Augment training pairs with rotations, flips, and color swaps.
ENABLE_DYNAMIC_AUGMENTATION = False    # On each training step, randomly shift grids within the 30x30 frame.
INFERENCE_AUG_MATCHES_TRAIN = True    # If True, inference uses the same augs as training. If False, it generates new random ones.

AUG_SETTINGS: Dict[str, Any] = {
    "aug_count": 1000,
    # For a 30x30 grid, how much padding can be added?
    # e.g., if a 5x5 grid is placed, it can be shifted by up to (30-5) = 25 cells.
    # This is handled dynamically based on grid size.
    "ARCMaxGridSize": 30,
}

# Hyperparameters
HPARAMS: Dict[str, Any] = {
    "grid_size": 30,
    "n_classes": 11,
    "in_channels": 20, # 11 for color one-hot, 9 for hidden state
    "hidden_channels": 9,
    "nn_hidden_dim": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_iterations": 5000,
    "prediction_steps": 30,
    "train_steps_min": 30,
    "train_steps_max": 30,
    "max_norm": 1.0
}

# --- 2.a. Visualisation Settings ---
VIZ_SETTINGS: Dict[str, Any] = {
    "max_aug_train_pairs": 8,                 # Max augmented train pairs to show
    "max_test_examples_per_case": 12,        # Max augmented test examples per test case to show
    "top_k_inverse_outputs": 6               # Max top inverse outputs to visualize per test case
}

# --- 2.b. Visualisation Helpers ---
def _get_arc_cmap():
    """Returns matplotlib cmap and norm for ARC colors with optional padding (-1)."""
    ARC_COLORS = [
        (0, 0, 0),                    # 0: black
        (0, 116/255, 217/255),        # 1: blue
        (255/255, 65/255, 54/255),    # 2: red
        (46/255, 204/255, 64/255),    # 3: green
        (255/255, 220/255, 0/255),    # 4: yellow
        (170/255, 170/255, 170/255),  # 5: grey
        (240/255, 22/255, 230/255),   # 6: magenta/pink
        (255/255, 133/255, 27/255),   # 7: orange
        (127/255, 219/255, 255/255),  # 8: cyan
        (135/255, 15/255, 35/255),    # 9: dark red
    ]
    PADDING_COLOR = (1.0, 1.0, 1.0)
    cmap_colors = [PADDING_COLOR] + ARC_COLORS
    cmap = ListedColormap(cmap_colors)
    bounds = list(range(len(cmap_colors) + 1))
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def _plot_grid(ax, grid: List[List[int]], title: str, cmap, norm):
    if grid is None or len(grid) == 0 or len(grid[0]) == 0:
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    # Shift values by +1 so -1 maps to 0, 0->1, ... 9->10
    shifted = [[cell + 1 for cell in row] for row in grid]
    data = np.array(shifted, dtype=np.int8)
    ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
    rows, cols = data.shape
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def visualise_task_augmentations(
    task_id: str,
    train_pairs: List[Dict[str, List[List[int]]]],
    augmented_train_pairs: List[List[Dict[str, List[List[int]]]]],
    test_inputs: List[Dict[str, List[List[int]]]],
    inference_records: List[Dict[str, Any]],
    output_dir: str,
):
    """
    Simple visualisation for a single task showing:
    - initial train pairs
    - augmented train pairs (sampled)
    - augmented test inputs, their generated raw outputs, and inverse-transformed outputs (sampled)
    - most common inverse transformed test outputs
    Saves a PDF named visualization_{task_id}.pdf under output_dir.
    """
    cmap, norm = _get_arc_cmap()
    pdf_path = os.path.join(output_dir, f"visualization_{task_id}.pdf")
    pdf = PdfPages(pdf_path)

    # Title page
    fig = plt.figure(figsize=(8, 11))
    fig.text(0.5, 0.5, f"Task {task_id}\nAugmentation Visualisation", ha='center', va='center', fontsize=20, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

    # 1) Initial Train Pairs (visualise actual model inputs/targets as 30x30)
    for idx, pair in enumerate(train_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        model_inp = to_model_input_grid2d(pair.get('input', []), HPARAMS['grid_size'], padding_value=-1)
        model_out = to_model_input_grid2d(pair.get('output', []), HPARAMS['grid_size'], padding_value=-1)
        _plot_grid(axes[0], model_inp, "Model Train Input (30x30)", cmap, norm)
        _plot_grid(axes[1], model_out, "Model Train Target (30x30)", cmap, norm)
        plt.suptitle(f"Initial Train Pair #{idx+1}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    # 2) Augmented Train Pairs (sample)
    # Flatten all augmented sets (skip first original which is also present in train_pairs)
    flat_aug_pairs = []
    for aug_set_idx, aug_set in enumerate(augmented_train_pairs):
        if aug_set_idx == 0:
            continue
        flat_aug_pairs.extend(aug_set)
    sampled_aug_pairs = flat_aug_pairs[:VIZ_SETTINGS["max_aug_train_pairs"]]
    for idx, pair in enumerate(sampled_aug_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        model_inp = to_model_input_grid2d(pair.get('input', []), HPARAMS['grid_size'], padding_value=-1)
        model_out = to_model_input_grid2d(pair.get('output', []), HPARAMS['grid_size'], padding_value=-1)
        _plot_grid(axes[0], model_inp, "Model Aug Train Input (30x30)", cmap, norm)
        _plot_grid(axes[1], model_out, "Model Aug Train Target (30x30)", cmap, norm)
        plt.suptitle(f"Augmented Train Pair Sample #{idx+1}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    # 3) Augmented Test Inputs and Outputs (per test case)
    for test_idx, rec in enumerate(inference_records):
        examples = rec.get('examples', [])
        if not examples:
            continue
        # Limit displayed examples
        examples = examples[:VIZ_SETTINGS["max_test_examples_per_case"]]
        for ex_idx, ex in enumerate(examples):
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
            _plot_grid(axes[0], ex.get('model_input', []), "Model Test Input (30x30)", cmap, norm)
            _plot_grid(axes[1], ex.get('raw_output', []), "Generated Raw Output", cmap, norm)
            _plot_grid(axes[2], ex.get('inverse_output', []), "Inverse-Transformed Output", cmap, norm)
            plt.suptitle(f"Test #{test_idx+1} • Example #{ex_idx+1}", fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

        # 4) Most common inverse-transformed outputs
        top_list = rec.get('top_inverse_outputs', [])
        if top_list:
            # Arrange up to 6 in 2x3 grid
            k = min(VIZ_SETTINGS["top_k_inverse_outputs"], len(top_list))
            rows = 2
            cols = 3
            fig, axes = plt.subplots(rows, cols, figsize=(9, 6))
            axes = axes.flatten()
            for i in range(cols * rows):
                ax = axes[i]
                if i < k:
                    grid_i, count_i = top_list[i]
                    _plot_grid(ax, grid_i, f"Top {i+1} (count={count_i})", cmap, norm)
                else:
                    ax.axis('off')
            plt.suptitle(f"Test #{test_idx+1} • Most Common Inverse Outputs", fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()
    print(f"  Visualisation for task {task_id} saved to {pdf_path}")

# --- 2. The CellularNN Model ---

class CellularNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, nn_hidden_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        
        # For a 3x3 neighborhood, there are 8 neighbors.
        perception_channels = self.in_channels + (8 * self.in_channels) 
        
        self.fc1 = nn.Conv2d(perception_channels, nn_hidden_dim, 1)
        self.fc2 = nn.Conv2d(nn_hidden_dim, self.in_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def get_neighbor_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alternative implementation to get neighbor states using padding and slicing,
        aiming to avoid F.unfold and its potential MPS fallback for 'col2im'.
        """
        padded_state = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
        B, C_state, H, W = x.shape

        neighbor_tensors = []

        # Iterate through the 3x3 neighborhood offsets, skipping the center (0,0)
        for r_offset in range(-1, 2):  # -1, 0, 1
            for c_offset in range(-1, 2):  # -1, 0, 1
                if r_offset == 0 and c_offset == 0:
                    continue  # Skip the center cell itself

                start_row = r_offset + 1
                end_row = start_row + H
                start_col = c_offset + 1
                end_col = start_col + W

                neighbor_slice = padded_state[:, :, start_row:end_row, start_col:end_col]
                neighbor_tensors.append(neighbor_slice)

        all_neighbors = torch.cat(neighbor_tensors, dim=1)
        return all_neighbors


    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        neighbor_channels = self.get_neighbor_states(x)
        return torch.cat([x, neighbor_channels], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        perception = self.perceive(x)
        h = F.relu(self.fc1(perception))
        dx = self.fc2(h)
        new_state = x + dx
        return new_state

# --- 3. Helper Functions ---
def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else: # Should not happen
        return arr

def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Applies the inverse of a dihedral transformation."""
    # Index corresponds to the original tid, and the value is its inverse.
    DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])

def generate_color_permutation_preserve_zero() -> np.ndarray:
    """Generate a color mapping that preserves 0 and shuffles 1..9."""
    perm_rest = np.random.permutation(np.arange(1, 10))
    mapping = np.empty(10, dtype=np.int32)
    mapping[0] = 0
    mapping[1:] = perm_rest
    return mapping

def apply_color_map_preserve_padding(grid: np.ndarray, color_map: np.ndarray, padding_value: int = -1) -> np.ndarray:
    """Apply color_map to grid values in [0..9], leaving padding_value untouched."""
    result = grid.copy()
    mask = result != padding_value
    result[mask] = color_map[result[mask]]
    return result

def to_model_input_grid2d(small_grid: np.ndarray, grid_size: int, padding_value: int = -1) -> List[List[int]]:
    """Convert a small 2D grid to a grid_size x grid_size 2D grid with padding_value."""
    if isinstance(small_grid, list):
        small_grid_np = np.array(small_grid, dtype=np.int32)
    else:
        small_grid_np = small_grid
    model_grid = np.full((grid_size, grid_size), padding_value, dtype=np.int32)
    h, w = small_grid_np.shape
    h = min(h, grid_size)
    w = min(w, grid_size)
    model_grid[:h, :w] = small_grid_np[:h, :w]
    return model_grid.tolist()

def apply_translational_augment(grid: np.ndarray, max_grid_size: int, padding_value: int = -1) -> np.ndarray:
    """Pads a small grid into a larger one with random top-left padding using padding_value (-1)."""
    padded_grid = np.full((max_grid_size, max_grid_size), fill_value=padding_value, dtype=grid.dtype)
    h, w = grid.shape

    if h >= max_grid_size or w >= max_grid_size:
        return grid[:max_grid_size, :max_grid_size]

    pad_r = np.random.randint(0, max_grid_size - h + 1)
    pad_c = np.random.randint(0, max_grid_size - w + 1)
    padded_grid[pad_r:pad_r+h, pad_c:pad_c+w] = grid
    return padded_grid

def create_array_from_grid(
    small_grid: List[List[int]], grid_size: int, in_channels: int, n_classes: int
) -> np.ndarray:
    """Creates a (grid_size, grid_size, in_channels) numpy array from a small grid."""
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    # Channel 0 is the "empty" channel
    arr[:, :, 0] = 1.0

    # Handle potential numpy conversion from list of lists of varying length
    if isinstance(small_grid, list):
        small_grid_np = np.array(small_grid, dtype=np.int32)
    else: # Already a numpy array
        small_grid_np = small_grid
    rows, cols = small_grid_np.shape
    max_rows, max_cols = min(rows, grid_size), min(cols, grid_size)

    for i in range(max_rows):
        for j in range(max_cols):
            pixel_val = small_grid_np[i, j]
            # Colors 0-9 map to channels 1-10
            if 0 <= pixel_val <= (n_classes - 2):
                arr[i, j, :n_classes] = 0.0
                arr[i, j, pixel_val + 1] = 1.0
    return arr

def tensor_to_grid(state_tensor: torch.Tensor, n_classes: int) -> List[List[int]]:
    """Converts a single predicted state tensor [C, H, W] to a List[List[int]] grid."""
    pred_indices = state_tensor.cpu()[:n_classes, :, :].argmax(dim=0).numpy()
    # Convert channel indices back to ARC color values. Channel 0 is empty, Channel 1 is color 0, etc.
    grid = (pred_indices - 1).tolist()
    # Replace -1 (from channel 0) with 0.
    # return [[max(0, cell) for cell in row] for row in grid]
    return grid

def depad_grid(grid: List[List[int]], padding_value: int = -1) -> List[List[int]]:
    """Removes padding from a grid by finding the smallest bounding box containing non-padding values."""
    if not grid or not grid[0]:
        return [[padding_value]]

    rows = len(grid)
    cols = len(grid[0])
    min_r, max_r, min_c, max_c = -1, -1, cols, -1
    found_non_padding = False

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx][c_idx] != padding_value:
                if not found_non_padding:
                    min_r = r_idx
                max_r = r_idx
                min_c = min(min_c, c_idx)
                max_c = max(max_c, c_idx)
                found_non_padding = True
    
    if not found_non_padding:
        return [[padding_value]]

    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]

# --- 4. The "Workhorse" Function ---

def train_and_predict_for_task(
    task_id: str, 
    train_pairs: List[Dict], 
    test_inputs: List[Dict], 
    device: torch.device, 
    hparams: Dict[str, Any]
) -> List[List[int]]:
    """
    Initializes, trains, and uses an NCA model for a single task.
    Returns a list of predicted grids for the test inputs.
    """
    # 1. Model & Optimizer Initialization
    model = CellularNN(
        hparams['in_channels'], hparams['n_classes'], hparams['nn_hidden_dim']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay']
    )

    # 2. Data Preparation with Augmentations
    augmented_train_pairs = []
    augmentation_transforms = [] # List of (trans_id, color_map) tuples

    if ENABLE_STATIC_AUGMENTATION:
        print(f"  Generating {AUG_SETTINGS['aug_count']} static augmentations...")
        # Add the original, untransformed pairs first
        augmented_train_pairs.append(train_pairs)
        augmentation_transforms.append(None) # Sentinel for original

        for _ in range(AUG_SETTINGS['aug_count']):
            trans_id = np.random.randint(0, 8)
            # Permute colors 1-9 only; preserve 0 (black)
            color_map = generate_color_permutation_preserve_zero()

            new_aug_pairs = []
            for pair in train_pairs:
                input_grid = np.array(pair['input'])
                output_grid = np.array(pair['output'])

                mapped_inp = apply_color_map_preserve_padding(input_grid, color_map, padding_value=-1)
                mapped_out = apply_color_map_preserve_padding(output_grid, color_map, padding_value=-1)
                aug_input = dihedral_transform(mapped_inp, trans_id)
                aug_output = dihedral_transform(mapped_out, trans_id)

                new_aug_pairs.append({'input': aug_input.tolist(), 'output': aug_output.tolist()})
            
            augmented_train_pairs.append(new_aug_pairs)
            augmentation_transforms.append((trans_id, color_map))
    else:
        # If not augmenting, just use the original pairs
        augmented_train_pairs.append(train_pairs)
        augmentation_transforms.append(None)
    
    grid_args = (hparams['grid_size'], hparams['in_channels'], hparams['n_classes'])

    def prepare_batch(current_aug_pairs: List[Dict]):
        """Helper to convert pairs to tensors, with optional dynamic augmentation."""
        input_list, target_list = [], []
        for pair in current_aug_pairs:
            input_grid = np.array(pair['input'])
            target_grid = np.array(pair['output'])
            
            if ENABLE_DYNAMIC_AUGMENTATION:
                # Apply random shift *before* converting to one-hot tensor
                input_grid = apply_translational_augment(input_grid, AUG_SETTINGS['ARCMaxGridSize'])
                target_grid = apply_translational_augment(target_grid, AUG_SETTINGS['ARCMaxGridSize'])

            input_list.append(
                torch.tensor(create_array_from_grid(input_grid, *grid_args)).permute(2, 0, 1)
            )
            target_list.append(
                torch.tensor(create_array_from_grid(target_grid, *grid_args)).permute(2, 0, 1)
            )

        return torch.stack(input_list).to(device), torch.stack(target_list).to(device)

    # 3. Training Loop
    model.train()
    for i in range(hparams['num_iterations']):
        # Select a random augmentation set for this training step
        # This is the "outer batch" over augmentations
        aug_idx = np.random.randint(len(augmented_train_pairs))
        current_aug_pairs = augmented_train_pairs[aug_idx]

        # Prepare the tensor batch, applying dynamic aug if enabled
        inp_batch, target_batch = prepare_batch(current_aug_pairs)

        optimizer.zero_grad()

        # NCA update steps
        state = inp_batch
        num_steps = np.random.randint(hparams['train_steps_min'], hparams['train_steps_max'] + 1)
        for _ in range(num_steps):
            state = model(state)
        
        # Loss calculation: MSE on all channels, with target being one-hot encoded.
        loss = F.mse_loss(state, target_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['max_norm'])
        optimizer.step()

        if i % 200 == 0:
            print(f"  Task {task_id}, Iter {i:04d}: Loss={loss.item():.4f}")
            
    # 4. Save Final Checkpoint
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{task_id}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"  Saved final model for task {task_id} to {final_checkpoint_path}")

    # 5. Prediction Phase
    model.eval()
    predicted_grids = []
    inference_records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for test_case in test_inputs:
            test_input_grid = test_case['input']
            record: Dict[str, Any] = {"examples": [], "top_inverse_outputs": []}
            max_examples = VIZ_SETTINGS["max_test_examples_per_case"]
            # If not augmenting, run standard prediction
            if not ENABLE_STATIC_AUGMENTATION:
                inp_array = create_array_from_grid(test_input_grid, *grid_args)
                state = torch.tensor(inp_array).permute(2, 0, 1).unsqueeze(0).to(device)
                for _ in range(hparams['prediction_steps']):
                    state = model(state)
                grid = tensor_to_grid(state.squeeze(0), hparams['n_classes'])
                depadded = depad_grid(grid)
                final_grid = [[max(0, cell) for cell in row] for row in depadded]
                # Append the two attempts for this specific test case
                predicted_grids.append([final_grid, final_grid])

            # --- Ensemble Inference with Augmentations ---
            print("  Running ensemble inference...")
            predictions_with_counts = collections.defaultdict(int)
            
            inference_transforms = augmentation_transforms
            if not INFERENCE_AUG_MATCHES_TRAIN:
                # Generate new random transforms for inference
                print("  Generating new random transforms for inference...")
                inference_transforms = [None] # Keep original
                for _ in range(AUG_SETTINGS['aug_count']):
                    trans_id = np.random.randint(0, 8)
                    color_map = generate_color_permutation_preserve_zero()
                    inference_transforms.append((trans_id, color_map))

            for transform in inference_transforms:
                current_test_input = np.array(test_input_grid)

                # 1. Augment the test input
                if transform is not None:
                    trans_id, color_map = transform
                    mapped = apply_color_map_preserve_padding(current_test_input, color_map, padding_value=-1)
                    current_test_input = dihedral_transform(mapped, trans_id)

                # NOTE: No dynamic (translational) augmentation during inference
                inp_array = create_array_from_grid(current_test_input.tolist(), *grid_args)
                state = torch.tensor(inp_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 2. Run the model
                for _ in range(hparams['prediction_steps']):
                    state = model(state)
                
                # 3. Get raw prediction grid (before inverse transforms)
                pred_grid = np.array(tensor_to_grid(state.squeeze(0), hparams['n_classes']))
                raw_pred_grid = pred_grid.copy()
                
                if transform is not None:
                    trans_id, color_map = transform
                    # Inverse color map
                    inv_color_map = np.argsort(color_map)
                    pred_grid = apply_color_map_preserve_padding(pred_grid, inv_color_map, padding_value=-1)
                    # Inverse dihedral transform
                    pred_grid = inverse_dihedral_transform(pred_grid, trans_id)
                
                # 4. Count the result
                # Convert to tuple of tuples to be hashable for the dictionary key
                depadded = depad_grid(pred_grid.tolist())
                final_grid = tuple(tuple(max(0, cell) for cell in row) for row in depadded)
                predictions_with_counts[final_grid] += 1

                # Collect a sample for visualisation
                if len(record["examples"]) < max_examples:
                    aug_input_grid = current_test_input.tolist()
                    # Store the actual 30x30 model input for visualization
                    model_input_grid = to_model_input_grid2d(aug_input_grid, HPARAMS['grid_size'], padding_value=-1)
                    # Keep raw output as-is (no inverse), displayable with cmap
                    raw_output_grid = raw_pred_grid.tolist()
                    inverse_output_grid = [list(row) for row in final_grid]
                    record["examples"].append({
                        "model_input": model_input_grid,
                        "raw_output": raw_output_grid,
                        "inverse_output": inverse_output_grid,
                    })
            
            # Find the top K most common predictions
            sorted_preds = sorted(predictions_with_counts.items(), key=lambda item: item[1], reverse=True)
            attempt1 = [list(row) for row in sorted_preds[0][0]]
            attempt2 = [list(row) for row in sorted_preds[1][0]] if len(sorted_preds) > 1 else attempt1
            predicted_grids.append([attempt1, attempt2])

            # Save top inverse outputs for visualisation
            top_k = min(VIZ_SETTINGS["top_k_inverse_outputs"], len(sorted_preds))
            record["top_inverse_outputs"] = [
                ([list(r) for r in key], cnt) for (key, cnt) in sorted_preds[:top_k]
            ]
            inference_records.append(record)

    # Visualise per-task if enabled
    if VISUALISE:
        visualise_task_augmentations(
            task_id=task_id,
            train_pairs=train_pairs,
            augmented_train_pairs=augmented_train_pairs,
            test_inputs=test_inputs,
            inference_records=inference_records,
            output_dir=OUTPUT_DIR,
        )

    return predicted_grids # Return predictions for ALL test cases

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    script_start_time = time.time()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    try:
        with open(INPUT_JSON_FILE, 'r') as f:
            challenges = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Input JSON not found at {INPUT_JSON_FILE}")
        print("Please check the `ARC_DATA_DIR` and `INPUT_JSON_FILE` variables.")
        exit()
    
    submission = {}

    # Main processing loop
    task_ids = list(challenges.keys())
    print(f"Found {len(task_ids)} tasks in {INPUT_JSON_FILE}")

    for i, task_id in enumerate(task_ids):
        task_data = challenges[task_id]
        print(f"\n--- Processing task {i+1}/{len(task_ids)}: {task_id} ---")

        train_pairs = task_data['train']
        # The 'test' field in training files are pairs, but in test files are just inputs.
        # We handle this by only looking at the 'input' key.
        test_inputs = task_data['test']

        if not train_pairs:
            print(f"  Skipping task {task_id}: No training pairs.")
            # Create default predictions for all test cases for this task, matching the expected format
            all_test_case_preds = [ [[0]], [[0]] ] * len(test_inputs)
        else:
            # The magic happens here
            all_test_case_preds = train_and_predict_for_task(
                task_id, 
                train_pairs, 
                test_inputs, 
                device, 
                HPARAMS
            )

        # Format for submission.json
        formatted_predictions = []
        # all_test_case_preds is now a list of [attempt1, attempt2] pairs
        for attempts in all_test_case_preds:
            attempt1, attempt2 = attempts[0], attempts[1]
            formatted_predictions.append({"attempt_1": attempt1, "attempt_2": attempt2})

        submission[task_id] = formatted_predictions # Assign list of prediction dicts

    # Save final submission file
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f)

    if VISUALISE:
        print("\nPer-task visualisations were generated alongside inference.")

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    print("\n-----------------------------------------")
    print(f"Success! Submission file saved to {SUBMISSION_FILE}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("-----------------------------------------")