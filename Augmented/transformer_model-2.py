import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Constants ---
# Feature flags
ENABLE_DYNAMIC_TRANSLATION = True
ENABLE_PADDING = True  # When False: no PAD/EOS, vocab becomes 10 (colors 0-9)
ENABLE_STATIC_AUGMENTATIONS = True

MAX_GRID_SIZE = 30

VOCAB_SIZE = 12 if ENABLE_PADDING else 10
COLOR_OFFSET = 2 if ENABLE_PADDING else 0

SEQ_LEN = MAX_GRID_SIZE * MAX_GRID_SIZE
NUM_AUGMENTATIONS = 30
ARCAugmentRetriesFactor = 5

# Global/static augmentations shared between training and inference
STATIC_AUGS: list[tuple[int, np.ndarray]] | None = None  # [(dihedral_id, color_map[0..9])]

# --- Config ---
file_path = os.path.join(".", "challenges.json")
Task_id = "7666fa5d"

Train = True
Epochs = 100
Batch_size = 64
Learning_rate = 1e-3
RNN_iterations = 1
TBPTT_WINDOW = 1  # Truncate BPTT across recurrence steps; gradients span at most this many steps
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

# Precompute inverse ids for the 8 dihedral transforms (computed after definition)
def _compute_dihedral_inverse_map() -> dict[int, int]:
    x = np.arange(12).reshape(3, 4)
    inv = {}
    for a in range(8):
        xa = dihedral_transform(x, a)
        for b in range(8):
            xb = dihedral_transform(xa, b)
            if np.array_equal(xb, x):
                inv[a] = b
                break
    return inv

DIHEDRAL_INV_MAP = _compute_dihedral_inverse_map()

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
    def __init__(self, augmented_pairs):
        self.augmented_pairs = augmented_pairs

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

# --- Recurrent non-autoregressive Transformer (with RoPE) ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000, max_len: int = SEQ_LEN):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int, device=None, dtype=None):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [b, s, h, d], cos/sin: [s, d]
    return (x * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(x) * sin.unsqueeze(0).unsqueeze(2))

class SelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, nheads: int):
        super().__init__()
        assert d_model % nheads == 0
        self.d_model = d_model
        self.nheads = nheads
        self.head_dim = d_model // nheads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.nheads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.nheads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.nheads, self.head_dim)
        cos, sin = self.rope(s, device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # attention
        attn_scores = torch.einsum('bsid,bsjd->bsij', q, k) / (self.head_dim ** 0.5)
        attn = attn_scores.softmax(dim=-1)
        ctx = torch.einsum('bsij,bsjd->bsid', attn, v)
        ctx = ctx.reshape(b, s, d)
        return self.o_proj(ctx)

class TransformerBlockRoPE(nn.Module):
    def __init__(self, d_model: int, nheads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttentionRoPE(d_model, nheads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class RecurrentTransformerNA(nn.Module):
    def __init__(self, d_model: int = 512, steps: int = 16, nheads: int = 8, nlayers: int = 8):
        super().__init__()
        self.steps = steps
        self.d_model = d_model
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embedding = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.ModuleList([TransformerBlockRoPE(d_model, nheads) for _ in range(nlayers)])
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        init = torch.empty(1, SEQ_LEN, d_model)
        nn.init.trunc_normal_(init, std=0.02)
        self.register_buffer("init_out_prev", init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        pos = torch.arange(s, device=x.device)
        fixed_inp = self.token_embedding(x) + self.pos_embedding(pos).unsqueeze(0)
        out_prev = self.init_out_prev[:, :s, :].expand(b, -1, -1)
        # Truncated BPTT over the recurrent "steps" dimension.
        # We unroll TBPTT_WINDOW steps at a time and detach the state between chunks
        # so gradients are truncated to at most TBPTT_WINDOW steps.
        window = TBPTT_WINDOW if (isinstance(TBPTT_WINDOW, int) and TBPTT_WINDOW > 0) else self.steps
        steps_remaining = self.steps
        while steps_remaining > 0:
            chunk = min(window, steps_remaining)
            for _ in range(chunk):
                y = out_prev + fixed_inp
                for blk in self.blocks:
                    y = blk(y)
                out_prev = y
            steps_remaining -= chunk
            # Detach to truncate gradient flow across recurrence chunks (except after final chunk)
            if steps_remaining > 0:
                out_prev = out_prev.detach()
        return self.head(out_prev)

# --- Data Loading / Augmentation ---

def _get_or_create_static_augmentations(num_augmentations: int) -> list[tuple[int, np.ndarray]]:
    global STATIC_AUGS
    if STATIC_AUGS is not None and len(STATIC_AUGS) == num_augmentations:
        return STATIC_AUGS
    augs: list[tuple[int, np.ndarray]] = []
    for _ in range(num_augmentations):
        tid = random.randint(0, 7)
        cmap = np.concatenate(([0], np.random.permutation(np.arange(1, 10))))
        augs.append((tid, cmap))
    STATIC_AUGS = augs
    return STATIC_AUGS

def load_dataset_for_task(task_id: str, json_path: str, num_augmentations: int = NUM_AUGMENTATIONS) -> ARCTaskDataset:
    with open(json_path, 'r') as f:
        all_tasks = json.load(f)
    task_data = all_tasks[task_id]
    train_pairs = task_data['train']

    print(f"Creating {1 + num_augmentations} augmentations for each of the {len(train_pairs)} training pairs...")
    augmented_data = []
    static_augs = _get_or_create_static_augmentations(num_augmentations) if ENABLE_STATIC_AUGMENTATIONS else []
    for pair in tqdm(train_pairs, desc="Augmenting Pairs"):
        inp, outp = np.array(pair['input']), np.array(pair['output'])
        augmented_data.append((inp, outp))  # original
        for trans_id, color_map in static_augs:
            aug_inp = dihedral_transform(color_map[inp], trans_id)
            aug_outp = dihedral_transform(color_map[outp], trans_id)
            augmented_data.append((aug_inp, aug_outp))

    return ARCTaskDataset(augmented_data)

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
def run_inference_on_test_inputs(model: nn.Module, task_id: str, json_path: str, device: torch.device) -> list[list[list[list[int]]]]:
    """Returns for each test grid the top-2 predictions after static aug voting."""
    with open(json_path, 'r') as f:
        all_tasks = json.load(f)
    task_data = all_tasks[task_id]
    test_items = task_data.get('test', [])

    static_augs = _get_or_create_static_augmentations(NUM_AUGMENTATIONS) if ENABLE_STATIC_AUGMENTATIONS else []
    results: list[list[list[list[int]]]] = []
    model.eval()

    def inverse_color_map_arr(cm: np.ndarray) -> np.ndarray:
        inv = np.zeros_like(cm)
        for i, v in enumerate(cm):
            inv[v] = i
        return inv

    for item in test_items:
        inp = np.array(item['input'], dtype=np.int64)
        h, w = inp.shape
        aug_batches: list[torch.Tensor] = []
        aug_meta: list[tuple[int, np.ndarray]] = []
        # Build augmented inputs
        for trans_id, cm in static_augs:
            aug_in = dihedral_transform(cm[inp], trans_id)
            toks = format_grid_dynamically(aug_in, pad_h=0, pad_w=0).unsqueeze(0)
            aug_batches.append(toks)
            aug_meta.append((trans_id, cm))
        if len(aug_batches) == 0:
            aug_batches.append(format_grid_dynamically(inp, pad_h=0, pad_w=0).unsqueeze(0))
            aug_meta.append((0, np.arange(10)))
        all_toks = torch.cat(aug_batches, dim=0).to(device)

        # Run in chunks to save memory
        preds: list[np.ndarray] = []
        bs = 64
        for i in range(0, all_toks.size(0), bs):
            logits = model(all_toks[i:i+bs])
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)

        # Invert transforms and vote
        counts: dict[tuple, int] = {}
        grids_cache: dict[tuple, list[list[int]]] = {}
        for k, pred_tokens in enumerate(preds):
            colors = np.clip(pred_tokens - COLOR_OFFSET, 0, 9) if COLOR_OFFSET > 0 else pred_tokens
            grid = colors.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE) if ENABLE_PADDING else colors.reshape(h, w)
            trans_id, cm = aug_meta[k]
            inv_cm = inverse_color_map_arr(cm)
            grid = inv_cm[grid]
            inv_tid = DIHEDRAL_INV_MAP[trans_id]
            grid = dihedral_transform(grid, inv_tid)
            if not ENABLE_PADDING:
                final = grid
            else:
                final = grid  # already aligned top-left as no dynamic translation
            key = tuple(final.flatten().tolist())
            counts[key] = counts.get(key, 0) + 1
            if key not in grids_cache:
                grids_cache[key] = final.astype(int).tolist()

        # pick top-2
        top2 = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]
        results.append([grids_cache[k] for k, _ in top2])

    return results

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

        model = RecurrentTransformerNA(d_model=512, steps=RNN_iterations, nheads=8, nlayers=8).to(device)
        train_model(model, dataloader, device, epochs=Epochs, learning_rate=Learning_rate, weight_decay=Weight_decay, max_norm=Max_norm)

        ckpt_name = f"tinyrnn_{Task_id}.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved checkpoint: {ckpt_name}")

        # Run inference on test inputs and save predictions
        predictions = run_inference_on_test_inputs(model, Task_id, file_path, device)
        out_path = os.path.join(".", f"inference_{Task_id}.json")
        with open(out_path, 'w') as f:
            json.dump({"task_id": Task_id, "predictions_top2": predictions}, f)
        print(f"Saved test predictions: {out_path}")

        # Visualize the last test input and its predicted output
        try:
            with open(file_path, 'r') as f:
                all_tasks = json.load(f)
            test_items = all_tasks[Task_id].get('test', [])
            if len(test_items) > 0:
                last_input_np = np.array(test_items[-1]['input'], dtype=np.int64)
                top2 = predictions[-1]
                # visualize top 2
                for i, pred_grid in enumerate(top2[:2]):
                    pred_np = np.array(pred_grid, dtype=np.int64)
                    if ENABLE_PADDING:
                        viz_input = last_input_np + COLOR_OFFSET
                        viz_output = pred_np + COLOR_OFFSET
                    else:
                        viz_input = last_input_np
                        viz_output = pred_np
                    visualize_datapoint(viz_input, viz_output, title=f"Task {Task_id}: Test Prediction Top-{i+1}")
        except Exception as e:
            print(f"Visualization failed: {e}")