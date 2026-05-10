import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Constants ---
# Feature flags
ENABLE_DYNAMIC_TRANSLATION = False
ENABLE_PADDING = False  # When False: no PAD/EOS, vocab becomes 10 (colors 0-9)
ENABLE_STATIC_AUGMENTATIONS = True

MAX_GRID_SIZE = 30

VOCAB_SIZE = 12 if ENABLE_PADDING else 10
COLOR_OFFSET = 2 if ENABLE_PADDING else 0

SEQ_LEN = MAX_GRID_SIZE * MAX_GRID_SIZE
NUM_AUGMENTATIONS = 1000
ARCAugmentRetriesFactor = 5

# --- Config ---
file_path = os.path.join(os.path.dirname(__file__), "tasks.json")
Task_id = "7666fa5d"

Train = True
Epochs = 100
Batch_size = 32
Learning_rate = 1e-3
RNN_iterations = 16
Weight_decay = 1e-4
Max_norm = 1.0

Visualize = False
Dataset_index = 100

# --- Helper Functions for Data ---

def dihedral_transform(grid: np.ndarray, tid: int) -> np.ndarray:
    """Applies one of 8 dihedral transformations."""
    if tid == 0: return grid
    if tid == 1: return np.rot90(grid, 1)
    if tid == 2: return np.rot90(grid, 2)
    if tid == 3: return np.rot90(grid, 3)
    if tid == 4: return np.fliplr(grid)
    if tid == 5: return np.flipud(grid)
    if tid == 6: return grid.T
    if tid == 7: return np.fliplr(np.rot90(grid, 1))
    return grid

def _compute_inverse_dihedral_map() -> dict[int, int]:
    """Precompute inverse transform id for each dihedral transform id [0..7]."""
    # Use a small asymmetric test grid to avoid accidental symmetries
    base = np.arange(1, 13, dtype=np.int64).reshape(3, 4)
    inverse_map: dict[int, int] = {}
    for tid in range(8):
        transformed = dihedral_transform(base, tid)
        for inv_tid in range(8):
            restored = dihedral_transform(transformed, inv_tid)
            if np.array_equal(restored, base):
                inverse_map[tid] = inv_tid
                break
    return inverse_map

INVERSE_DIHEDRAL_ID = _compute_inverse_dihedral_map()

def generate_static_transforms(num_augmentations: int) -> list[dict]:
    """Generate a global list of static transforms (dihedral + color_map).

    The first transform is identity. The rest are random but unique combos.
    This exact list will be applied to ALL train pairs and ALL test inputs.
    """
    transforms: list[dict] = []
    # Identity
    identity_color_map = np.arange(10, dtype=np.int64)
    transforms.append({"trans_id": 0, "color_map": identity_color_map})

    seen: set[tuple[int, bytes]] = {(0, identity_color_map.tobytes())}
    tries = 0
    target_total = 1 + max(0, num_augmentations)
    while len(transforms) < target_total and tries < ARCAugmentRetriesFactor * max(1, num_augmentations):
        tries += 1
        trans_id = random.randint(0, 7)
        color_map = np.concatenate(([0], np.random.permutation(np.arange(1, 10)))).astype(np.int64)
        key = (trans_id, color_map.tobytes())
        if key in seen:
            continue
        seen.add(key)
        transforms.append({"trans_id": trans_id, "color_map": color_map})

    return transforms

def apply_static_transform(grid: np.ndarray, trans: dict) -> np.ndarray:
    """Apply color and dihedral transform to a raw color grid (values 0..9)."""
    color_map: np.ndarray = trans["color_map"]
    trans_id: int = trans["trans_id"]
    colored = color_map[grid]
    return dihedral_transform(colored, trans_id)

def invert_static_transform_on_grid(grid: np.ndarray, trans: dict) -> np.ndarray:
    """Invert the static transform (color + dihedral) on a predicted color grid."""
    trans_id: int = trans["trans_id"]
    inv_trans_id = INVERSE_DIHEDRAL_ID[trans_id]
    inv_grid = dihedral_transform(grid, inv_trans_id)

    # Invert color map
    color_map: np.ndarray = trans["color_map"]
    inv_color_map = np.zeros_like(color_map)
    for i, v in enumerate(color_map):
        inv_color_map[v] = i
    inv_grid = inv_color_map[inv_grid]
    return inv_grid

def format_grid_dynamically(grid: np.ndarray, pad_h: int | None = None, pad_w: int | None = None) -> torch.Tensor:
    """Formats a grid for the model.

    - If padding is enabled: place onto 30x30 with optional translation and add EOS boundaries.
    - If padding is disabled: return flattened grid with no PAD/EOS; values are offset by COLOR_OFFSET.
    """
    grid = np.array(grid, dtype=np.int64)

    # No padding mode: return flattened values with appropriate offset
    if not ENABLE_PADDING:
        return torch.from_numpy((grid + COLOR_OFFSET).flatten())

    padded_grid = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int64)

    # Dynamic translation
    h, w = grid.shape
    if pad_h is None or pad_w is None:
        # If the grid already occupies the full canvas (30x30), do not translate.
        if h == MAX_GRID_SIZE and w == MAX_GRID_SIZE:
            pad_h, pad_w = 0, 0
        else:
            if ENABLE_DYNAMIC_TRANSLATION:
                pad_h = random.randint(0, MAX_GRID_SIZE - h)
                pad_w = random.randint(0, MAX_GRID_SIZE - w)
            else:
                pad_h, pad_w = 0, 0

    # Shift integers and place on canvas
    padded_grid[pad_h:pad_h+h, pad_w:pad_w+w] = grid + COLOR_OFFSET

    # Add EOS markers only if there is remaining space at the boundary
    if pad_h + h < MAX_GRID_SIZE:
        padded_grid[pad_h + h, pad_w:pad_w + w] = 1
    if pad_w + w < MAX_GRID_SIZE:
        padded_grid[pad_h:pad_h + h, pad_w + w] = 1

    return torch.from_numpy(padded_grid.flatten())

# --- Custom Dataset ---

class ARCTaskDataset(Dataset):
    def __init__(self, augmented_pairs, static_transforms: list[dict] | None = None):
        self.augmented_pairs = augmented_pairs
        self.static_transforms = static_transforms or []

    def __len__(self):
        return len(self.augmented_pairs)

    def __getitem__(self, idx):
        input_grid, output_grid = self.augmented_pairs[idx]
        # Apply the SAME translation to both input and output of the pair
        if ENABLE_PADDING:
            in_h, in_w = np.array(input_grid).shape
            out_h, out_w = np.array(output_grid).shape

            max_h = max(in_h, out_h)
            max_w = max(in_w, out_w)

            # Ensure both grids can be placed with the same offsets
            max_pad_h = MAX_GRID_SIZE - max_h
            max_pad_w = MAX_GRID_SIZE - max_w

            if ENABLE_DYNAMIC_TRANSLATION:
                pad_h = random.randint(0, max(0, max_pad_h))
                pad_w = random.randint(0, max(0, max_pad_w))
            else:
                pad_h = 0
                pad_w = 0

            input_tensor = format_grid_dynamically(input_grid, pad_h=pad_h, pad_w=pad_w)
            output_tensor = format_grid_dynamically(output_grid, pad_h=pad_h, pad_w=pad_w)
            return input_tensor, output_tensor
        else:
            # No padding/translation
            input_tensor = format_grid_dynamically(input_grid)
            output_tensor = format_grid_dynamically(output_grid)
            return input_tensor, output_tensor

# --- Model with RoPE ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(SEQ_LEN, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        return self.cos_cached[:x.shape[1]], self.sin_cached[:x.shape[1]]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class ResidualRNN(nn.Module):
    def __init__(self, embedding_dim: int = 64, steps: int = 3):
        super().__init__()
        self.steps = steps
        self.token_embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.rope = RotaryEmbedding(embedding_dim)
        self.rnn = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.head = nn.Linear(embedding_dim, VOCAB_SIZE)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(x)
        cos, sin = self.rope(embedded)
        s = apply_rotary_pos_emb(embedded, cos.unsqueeze(0), sin.unsqueeze(0))
        h = None
        for _ in range(self.steps):
            y, h = self.rnn(s, h)
            s = s + y
        return self.head(s)

# --- Data Loading / Augmentation ---

def load_dataset_for_task(task_id: str, json_path: str, num_augmentations: int = NUM_AUGMENTATIONS) -> ARCTaskDataset:
    with open(json_path, 'r') as f:
        all_tasks = json.load(f)
    task_data = all_tasks[task_id]
    train_pairs = task_data['train']

    print(f"Creating {1 + num_augmentations} augmentations for each of the {len(train_pairs)} training pairs...")
    # Create a single global list of static transforms used across train and test
    if ENABLE_STATIC_AUGMENTATIONS and num_augmentations > 0:
        static_transforms = generate_static_transforms(num_augmentations)
    else:
        static_transforms = generate_static_transforms(0)

    augmented_data = []
    for pair in tqdm(train_pairs, desc="Augmenting Pairs"):
        inp, outp = np.array(pair['input']), np.array(pair['output'])
        for trans in static_transforms:
            aug_inp = apply_static_transform(inp, trans)
            aug_outp = apply_static_transform(outp, trans)
            augmented_data.append((aug_inp, aug_outp))

    return ARCTaskDataset(augmented_data, static_transforms=static_transforms)

# --- Training ---

def train_model(model: nn.Module, dataloader: DataLoader, device: torch.device, epochs: int = 3, learning_rate: float = 1e-3, weight_decay: float = Weight_decay, max_norm: float = Max_norm) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

# --- Inference ---

@torch.no_grad()
def run_inference_on_test_inputs(model: nn.Module, task_id: str, json_path: str, device: torch.device, static_transforms: list[dict]) -> list[list[list[list[int]]]]:
    """Run inference on test inputs with the SAME static augmentations used for training.

    For each test grid, generate 1+NUM_AUGMENTATIONS transformed inputs (no dynamic translation).
    Run them through the model, invert the transforms on outputs, and vote by most common.
    Returns, for each test input, the top-2 predicted grids (list of up to 2 grids).
    """
    with open(json_path, 'r') as f:
        all_tasks = json.load(f)
    task_data = all_tasks[task_id]
    test_items = task_data.get('test', [])

    top2_predictions: list[list[list[list[int]]]] = []
    model.eval()

    for item in test_items:
        input_grid_raw = np.array(item['input'], dtype=np.int64)

        # Collect predictions for each transform, invert them, and vote
        counts: dict[bytes, tuple[np.ndarray, int]] = {}

        for trans in static_transforms:
            # Apply the same static transform as used in training
            transformed_grid = apply_static_transform(input_grid_raw, trans)

            # Format deterministically for test: no dynamic translation, fixed top-left if padding
            if ENABLE_PADDING:
                input_tokens = format_grid_dynamically(transformed_grid, pad_h=0, pad_w=0).unsqueeze(0).to(device)
            else:
                input_tokens = format_grid_dynamically(transformed_grid).unsqueeze(0).to(device)

            # Model forward
            logits = model(input_tokens)
            pred_tokens = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy()

            # Map back to colors 0..9
            if COLOR_OFFSET > 0:
                color_np = np.clip(pred_tokens - COLOR_OFFSET, 0, 9)
            else:
                color_np = pred_tokens

            # Reshape to grid
            if ENABLE_PADDING:
                pred_grid = color_np.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
            else:
                ph, pw = transformed_grid.shape
                pred_grid = color_np.reshape(ph, pw)

            # Invert transform to original orientation/colors
            inv_grid = invert_static_transform_on_grid(pred_grid, trans)

            # Vote
            key = inv_grid.tobytes()
            if key in counts:
                existing_grid, cnt = counts[key]
                counts[key] = (existing_grid, cnt + 1)
            else:
                counts[key] = (inv_grid, 1)

        # Rank by count desc
        ranked = sorted(counts.values(), key=lambda x: x[1], reverse=True)
        top_grids = [ranked[i][0].astype(int).tolist() for i in range(min(2, len(ranked)))]
        top2_predictions.append(top_grids)

    return top2_predictions

# --- Visualisation functions ---

def visualize_datapoint(input_grid, output_grid, title="ARC Task Visualization"):
    """
    Visualizes an input/output grid pair from an ARC task.
    Can handle both raw numpy arrays and formatted 1D Tensors.
    """
    # --- Data Preprocessing ---
    # If the input is a flattened tensor, reshape it to a 2D grid when padding is enabled
    if isinstance(input_grid, torch.Tensor) and input_grid.ndim == 1:
        if ENABLE_PADDING:
            input_grid = input_grid.view(MAX_GRID_SIZE, MAX_GRID_SIZE)
        else:
            raise ValueError("Cannot visualize 1D tensors without padding; original 2D shape is unknown.")
    if isinstance(output_grid, torch.Tensor) and output_grid.ndim == 1:
        if ENABLE_PADDING:
            output_grid = output_grid.view(MAX_GRID_SIZE, MAX_GRID_SIZE)
        else:
            raise ValueError("Cannot visualize 1D tensors without padding; original 2D shape is unknown.")
        
    # Convert tensors to numpy arrays for plotting
    if isinstance(input_grid, torch.Tensor):
        input_grid = input_grid.cpu().numpy()
    if isinstance(output_grid, torch.Tensor):
        output_grid = output_grid.cpu().numpy()

    # --- Colormap Definition ---
    # Define a color for each value in the vocabulary
    # 0:PAD, 1:EOS, 2-11:colors 0-9
    colors = [
        '#808080',  # 0: PAD - Gray
        '#FFFFFF',  # 1: EOS - White
        '#000000',  # 2: Color - Black
        '#0074D9',  # 3: Color 0 - Blue
        '#FF4136',  # 4: Color 1 - Red
        '#2ECC40',  # 5: Color 2 - Green
        '#FFDC00',  # 6: Color 3 - Yellow
        '#B10DC9',  # 7: Color 4 - Purple
        '#F012BE',  # 8: Color 5 - Pink 
        '#FF851B',  # 9: Color 6 - Orange
        '#7FDBFF',  # 10: Color 7 - Light Blue
        '#4c3228',  # 11: Color 8 - Brown
    ]
    if not ENABLE_PADDING:
        colors = colors[2:]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, VOCAB_SIZE, 1), cmap.N)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Plot Input
    axes[0].imshow(input_grid, cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title(f"Input ({input_grid.shape[0]}x{input_grid.shape[1]})")
    axes[0].set_xticks(np.arange(-.5, input_grid.shape[1], 1), minor=True)
    axes[0].set_yticks(np.arange(-.5, input_grid.shape[0], 1), minor=True)
    axes[0].grid(which="minor", color="w", linestyle='-', linewidth=1)
    axes[0].tick_params(which="minor", size=0)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])


    # Plot Output
    im = axes[1].imshow(output_grid, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title(f"Output ({output_grid.shape[0]}x{output_grid.shape[1]})")
    axes[1].set_xticks(np.arange(-.5, output_grid.shape[1], 1), minor=True)
    axes[1].set_yticks(np.arange(-.5, output_grid.shape[0], 1), minor=True)
    axes[1].grid(which="minor", color="w", linestyle='-', linewidth=1)
    axes[1].tick_params(which="minor", size=0)
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=np.arange(VOCAB_SIZE), orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.ax.set_xticklabels([f'{i}' for i in range(VOCAB_SIZE)])
    cbar.set_label("Vocabulary (0:PAD, 1:EOS, 2-11:Colors)" if ENABLE_PADDING else "Vocabulary (0-9: Colors)")


    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# --- Visualization Demonstration ---

def demonstrate_visualization(task_id, json_path, dataset_index: int = 0, num_augmentations: int = NUM_AUGMENTATIONS):
    print("\n--- Running Visualization Demonstration ---")

    # Build the full dataset using the same pipeline as training
    dataset = load_dataset_for_task(task_id, json_path, num_augmentations)
    total_examples = len(dataset.augmented_pairs)

    # Validate index
    if dataset_index < 0 or dataset_index >= total_examples:
        print(f"Requested index {dataset_index} is out of range [0, {total_examples - 1}]. Using 0 instead.")
        dataset_index = 0

    # 1) Visualize the augmented-but-unformatted pair (shows dihedral + color mapping)
    aug_input_np, aug_output_np = dataset.augmented_pairs[dataset_index]
    print(f"Visualizing augmented raw pair at index {dataset_index} (before padding/translation)...")
    visualize_datapoint(aug_input_np + COLOR_OFFSET, aug_output_np + COLOR_OFFSET, title=f"Task {task_id}: Augmented Raw Pair @ {dataset_index}")

    # 2) Visualize the formatted pair as consumed by the model (adds padding/translation/EOS)
    print("\nVisualizing the formatted datapoint (after padding/translation/EOS)...")
    if ENABLE_PADDING:
        formatted_input, formatted_output = dataset[dataset_index]
        visualize_datapoint(formatted_input, formatted_output, title=f"Task {task_id}: Formatted Datapoint @ {dataset_index}")
    else:
        print("Padding disabled: skipping visualization of formatted datapoint (original 2D shape unknown for flattened tensors).")


if __name__ == '__main__':

    if Visualize:
        demonstrate_visualization(Task_id, file_path, Dataset_index, NUM_AUGMENTATIONS)
    if Train:
        # Minimal training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")

        dataset = load_dataset_for_task(Task_id, file_path, num_augmentations=NUM_AUGMENTATIONS)
        dataloader = DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=0)

        model = ResidualRNN(embedding_dim=64, steps=RNN_iterations).to(device)
        train_model(model, dataloader, device, epochs=Epochs, learning_rate=Learning_rate, weight_decay=Weight_decay, max_norm=Max_norm)

        ckpt_name = f"tinyrnn_{Task_id}.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved checkpoint: {ckpt_name}")

        # Run inference on test inputs with the SAME static transforms and save top-2 predictions
        predictions_top2 = run_inference_on_test_inputs(model, Task_id, file_path, device, dataset.static_transforms)
        out_path = os.path.join(os.path.dirname(__file__), f"inference_{Task_id}.json")
        with open(out_path, 'w') as f:
            json.dump({"task_id": Task_id, "top2_predictions": predictions_top2}, f)
        print(f"Saved test predictions: {out_path}")

        # Visualize the last test input and the top-2 predicted outputs
        try:
            with open(file_path, 'r') as f:
                all_tasks = json.load(f)
            test_items = all_tasks[Task_id].get('test', [])
            if len(test_items) > 0:
                last_input_np = np.array(test_items[-1]['input'], dtype=np.int64)
                last_top_candidates = predictions_top2[-1] if len(predictions_top2) > 0 else []

                if ENABLE_PADDING:
                    viz_input = last_input_np + COLOR_OFFSET
                else:
                    viz_input = last_input_np
                # Visualize only the top 2
                for rank, cand in enumerate(last_top_candidates[:2], start=1):
                    cand_np = np.array(cand, dtype=np.int64)
                    if ENABLE_PADDING:
                        viz_output = cand_np + COLOR_OFFSET
                    else:
                        viz_output = cand_np
                    visualize_datapoint(viz_input, viz_output, title=f"Task {Task_id}: Test Prediction (Top-{rank})")
        except Exception as e:
            print(f"Visualization failed: {e}")