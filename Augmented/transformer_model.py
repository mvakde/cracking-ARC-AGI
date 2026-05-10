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
import torch.nn.functional as F
import hashlib

# --- Training Strategy ---
# This model is configured to train with transform-grouped batches:
# - Each batch contains all training pairs with the same transform applied
# - Set Batch_size to the number of training pairs for optimal grouping
# - Shuffle is disabled to maintain transform consistency within batches
# - Total examples = num_training_pairs * num_augmentations, ensuring complete batches

# --- Constants ---
MAX_GRID_SIZE = 30
VOCAB_SIZE = 12  # 0:PAD, 1:EOS, 2-11:Colors
COLOR_OFFSET = 2

IDENTITY_CM = np.arange(10, dtype=np.int64)

SEQ_LEN = MAX_GRID_SIZE * MAX_GRID_SIZE
NUM_AUGMENTATIONS = 30
ARCAugmentRetriesFactor = 5

# Global/static augmentations shared between training and inference
STATIC_AUGS: list[tuple[int, np.ndarray]] | None = None  # [(dihedral_id, color_map[0..9])]
STATIC_AUGS_TASK_ID: str | None = None

# --- Config ---
file_path = os.path.join(".", "challenges.json")
Task_id = "2697da3f"

Train = False
Epochs = 10
# IMPORTANT: Set Batch_size to the number of training pairs for optimal transform grouping
# Each batch will contain all training pairs with the same transform applied
# Example: if there are 4 training pairs, set Batch_size = 4
Batch_size = 3
Learning_rate = 1e-3
RNN_iterations = 4
TBPTT_WINDOW = 4  # Truncate BPTT across recurrence steps; gradients span at most this many steps
Weight_decay = 1e-4
Max_norm = 1.0

Visualize = True
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

def format_grid_with_padding(grid: np.ndarray, pad_h: int | None = None, pad_w: int | None = None) -> torch.Tensor:
    """Formats a grid for the model by placing it onto 30x30 with specified translation and adding EOS boundaries."""
    grid = np.array(grid, dtype=np.int64)
    padded_grid = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int64)

    # Use provided translation or default to top-left (0,0)
    h, w = grid.shape
    if pad_h is None:
        pad_h = 0
    if pad_w is None:
        pad_w = 0
    
    # Safety check: ensure the grid fits within bounds
    if pad_h + h > MAX_GRID_SIZE or pad_w + w > MAX_GRID_SIZE:
        raise ValueError(f"Grid placement would go out of bounds: grid {h}x{w} at position ({pad_h},{pad_w}) exceeds {MAX_GRID_SIZE}x{MAX_GRID_SIZE}")

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
        # Return preformatted tensors directly (static pipeline)
        return input_grid, output_grid

# --- Recurrent non-autoregressive Transformer (with RoPE) ---

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        return x * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: float = 2.6667, bias: bool = True):
        super().__init__()
        inner = int(round(expansion * d_model))
        self.wv = nn.Linear(d_model, inner, bias=bias)  # value path
        self.wg = nn.Linear(d_model, inner, bias=bias)  # gate path
        self.wo = nn.Linear(inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.wv(x)
        g = F.silu(self.wg(x))
        return self.wo(v * g)

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
        # swap LayerNorm → RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.attn = SelfAttentionRoPE(d_model, nheads)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, expansion=2.6667)  # was: Linear→GELU→Linear(4x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # POST-residual: sublayer, add residual, then norm
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
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
        fixed_inp = self.token_embedding(x) # + self.pos_embedding(pos).unsqueeze(0)
        # out_prev = self.init_out_prev[:, :s, :].expand(b, -1, -1)
        out_prev = fixed_inp
        # Truncated BPTT over the recurrent "steps" dimension.
        # We unroll TBPTT_WINDOW steps at a time and detach the state between chunks
        # so gradients are truncated to at most TBPTT_WINDOW steps.
        window = TBPTT_WINDOW if (isinstance(TBPTT_WINDOW, int) and TBPTT_WINDOW > 0) else self.steps
        steps_remaining = self.steps
        while steps_remaining > 0:
            chunk = min(window, steps_remaining)
            for _ in range(chunk):
                y = out_prev #+ fixed_inp
                for blk in self.blocks:
                    y = blk(y)
                out_prev = y
            steps_remaining -= chunk
            # Detach to truncate gradient flow across recurrence chunks (except after final chunk)
            if steps_remaining > 0:
                out_prev = out_prev.detach()
        return self.head(out_prev)

# --- Data Loading / Augmentation ---

def _select_unique_transforms(task_data: dict, num_augmentations: int) -> list[tuple[int, np.ndarray]]:
    """Select up to num_augmentations unique (dihedral, color_map) transforms
    such that the transformed cumulative set of grids (all train inputs/outputs and test inputs)
    is unique per transform.
    The total number of attempts is limited to ARCAugmentRetriesFactor * num_augmentations.
    """
    train_pairs = task_data.get('train', [])
    test_items = task_data.get('test', [])

    def signature_for_transform(trans_id: int, color_map: np.ndarray) -> str:
        hasher = hashlib.sha256()
        hasher.update(b"ARC_STATIC_AUG_V1")
        hasher.update(np.array([trans_id, MAX_GRID_SIZE], dtype=np.int64).tobytes())

        # Train pairs: include both input and output post-transform
        for pair in train_pairs:
            inp = np.array(pair['input'], dtype=np.int64)
            outp = np.array(pair['output'], dtype=np.int64)
            tin = dihedral_transform(color_map[inp], trans_id)
            tout = dihedral_transform(color_map[outp], trans_id)
            hasher.update(np.array(tin.shape, dtype=np.int64).tobytes())
            hasher.update(tin.astype(np.int16, copy=False).tobytes())
            hasher.update(np.array(tout.shape, dtype=np.int64).tobytes())
            hasher.update(tout.astype(np.int16, copy=False).tobytes())

        # Test inputs: include input post-transform
        for item in test_items:
            inp = np.array(item['input'], dtype=np.int64)
            tin = dihedral_transform(color_map[inp], trans_id)
            hasher.update(np.array(tin.shape, dtype=np.int64).tobytes())
            hasher.update(tin.astype(np.int16, copy=False).tobytes())

        return hasher.hexdigest()

    seen: set[str] = set()
    selected: list[tuple[int, np.ndarray]] = []
    max_tries = max(1, int(ARCAugmentRetriesFactor) * int(num_augmentations))
    tries = 0
    while len(selected) < num_augmentations and tries < max_tries:
        tries += 1
        tid = random.randint(0, 7)
        # Keep color 0 fixed as PAD; shuffle 1..9 as colors
        cmap = np.concatenate(([0], np.random.permutation(np.arange(1, 10))))
        sig = signature_for_transform(tid, cmap)
        if sig in seen:
            continue
        seen.add(sig)
        selected.append((tid, cmap))

    if len(selected) < num_augmentations:
        print(f"Warning: Selected only {len(selected)}/{num_augmentations} unique static transforms after {tries} attempts.")
    return selected

def _get_or_create_static_augmentations(task_id: str, task_data: dict, num_augmentations: int) -> list[tuple[int, np.ndarray]]:
    global STATIC_AUGS, STATIC_AUGS_TASK_ID
    if STATIC_AUGS is not None and STATIC_AUGS_TASK_ID == task_id and len(STATIC_AUGS) == num_augmentations:
        return STATIC_AUGS
    STATIC_AUGS = _select_unique_transforms(task_data, num_augmentations)
    STATIC_AUGS_TASK_ID = task_id
    return STATIC_AUGS

def load_dataset_for_task(task_id: str, json_path: str, num_augmentations: int = NUM_AUGMENTATIONS) -> ARCTaskDataset:
    with open(json_path, 'r') as f:
        all_tasks = json.load(f)
    task_data = all_tasks[task_id]
    train_pairs = task_data.get('train', [])


    augmented_examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    static_augs = _get_or_create_static_augmentations(task_id, task_data, num_augmentations)

    if not any((tid == 0 and np.array_equal(cm, IDENTITY_CM)) for tid, cm in static_augs):
        static_augs = [(0, IDENTITY_CM)] + static_augs

    # For each transform group, pad top-left and add EOS, then translate all but one pair randomly (static per dataset)
    # Data organization: [transform1_pair1, transform1_pair2, ..., transform1_pairN, transform2_pair1, transform2_pair2, ...]
    # This ensures that when batch_size = num_training_pairs, each batch contains the same transform applied to all pairs
    for (trans_id, color_map) in tqdm(static_augs, desc="Building Static Aug Groups"):
        if len(train_pairs) == 0:
            continue
        keep_idx = random.randint(0, len(train_pairs) - 1)
        for i, pair in enumerate(train_pairs):
            inp = np.array(pair['input'], dtype=np.int64)
            outp = np.array(pair['output'], dtype=np.int64)
            aug_inp = dihedral_transform(color_map[inp], trans_id)
            aug_outp = dihedral_transform(color_map[outp], trans_id)
            
            # Ensure both input and output have the same dimensions after transform
            # Note: dihedral transforms can change grid dimensions (e.g., transpose changes 6x4 to 4x6)
            h_in, w_in = aug_inp.shape
            h_out, w_out = aug_outp.shape
            
            # Use the larger dimensions to ensure both grids fit
            h, w = max(h_in, h_out), max(w_in, w_out)
            


            # Leave a 1-cell margin on bottom/right so EOS borders can be written when possible.
            # If h==MAX_GRID_SIZE or w==MAX_GRID_SIZE, margin is 0 (EOS border won't be possible).
            max_pad_h = max(0, MAX_GRID_SIZE - h - 1)
            max_pad_w = max(0, MAX_GRID_SIZE - w - 1)

            if i == keep_idx:
                pad_h, pad_w = 0, 0
            else:
                # Only apply random translation if there's actually space to move
                pad_h = random.randint(0, max_pad_h) if max_pad_h > 0 else 0
                pad_w = random.randint(0, max_pad_w) if max_pad_w > 0 else 0
               


            # Use the same padding for both input and output to ensure consistent dimensions
            input_tensor = format_grid_with_padding(aug_inp, pad_h=pad_h, pad_w=pad_w)
            output_tensor = format_grid_with_padding(aug_outp, pad_h=pad_h, pad_w=pad_w)
            augmented_examples.append((input_tensor, output_tensor))

    return ARCTaskDataset(augmented_examples)

# --- Training ---

def train_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                epochs: int = 3, learning_rate: float = 1e-3,
                weight_decay: float = 1e-4, max_norm: float = 1.0) -> None:
    # ---- compute inverse-frequency weights over colors 2..11, ignoring PAD/EOS ----
    with torch.no_grad():
        color_counts = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        ds = dataloader.dataset  # uses your ARCTaskDataset
        for _, tgt in ds:
            # tgt is a 1D tensor of length 900 (30x30) with tokens 0..11
            vals, cnts = torch.unique(tgt, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                if 2 <= v <= 11:  # ignore PAD=0 and EOS=1 in the stats
                    color_counts[v] += float(c)

        weights = torch.ones(VOCAB_SIZE, dtype=torch.float32)
        denom = color_counts[2:12].clone()
        denom[denom == 0] = 1.0  # avoid div-by-zero for unseen colors
        total = color_counts[2:12].sum().clamp_min(1.0)
        inv = total / denom                   # inverse freq
        inv = inv / inv.mean().clamp_min(1e-8)  # optional: normalize around 1.0
        # optional: cap extremes to prevent instability if distribution is very skewed
        inv = inv.clamp_max(10.0)

        weights[2:12] = inv
        # not strictly needed since we mask them out, but harmless to make explicit:
        weights[0] = 0.0  # PAD
        weights[1] = 0.0  # EOS
        weights = weights.to(device)

    # masked, per-token loss with weights
    criterion = nn.CrossEntropyLoss(reduction='none', weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)  # [B, S, V]
            B, S, V = logits.shape
            logits_f = logits.view(B*S, V)
            targets_f = targets.view(B*S)

            # ignore PAD=0 and EOS=1 by masking
            mask = (targets_f >= 2)  # keep only real colors
            per_tok_loss = criterion(logits_f, targets_f)           # [B*S]
            loss = (per_tok_loss * mask.float()).sum() / mask.sum().clamp_min(1)

            loss.backward()
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")



# --- Helper Functions for Inference ---

def inverse_color_map_arr(cm: np.ndarray) -> np.ndarray:
    inv = np.zeros_like(cm)
    for i, v in enumerate(cm):
        inv[v] = i
    return inv

def depad_tokens_to_content(tokens_grid: np.ndarray, min_h: int | None = None, min_w: int | None = None) -> np.ndarray:
    assert tokens_grid.shape == (MAX_GRID_SIZE, MAX_GRID_SIZE)

    eos_mask = (tokens_grid == 1)
    rows_with_eos = np.where(np.any(eos_mask, axis=1))[0]
    cols_with_eos = np.where(np.any(eos_mask, axis=0))[0]

    # 1) Prefer EOS borders if present AND we know the target size
    if rows_with_eos.size > 0 and cols_with_eos.size > 0 and min_h and min_w:
        r_border = int(rows_with_eos[0])  # bottom-of-content row (exclusive border)
        c_border = int(cols_with_eos[0])  # right-of-content col (exclusive border)
        r0 = max(0, r_border - min_h)
        c0 = max(0, c_border - min_w)
        r1 = min(MAX_GRID_SIZE, r_border)
        c1 = min(MAX_GRID_SIZE, c_border)
        if r1 <= r0 or c1 <= c0:
            raise ValueError("EOS-derived crop invalid")
        return tokens_grid[r0:r1, c0:c1]

    # 2) Fallback to bbox over colors (>=2)
    content_mask = (tokens_grid >= 2)
    if np.any(content_mask):
        r_any = np.any(content_mask, axis=1)
        c_any = np.any(content_mask, axis=0)
        r_indices = np.where(r_any)[0]
        c_indices = np.where(c_any)[0]
        r0 = int(r_indices[0]); r1 = int(r_indices[-1]) + 1
        c0 = int(c_indices[0]); c1 = int(c_indices[-1]) + 1
        return tokens_grid[r0:r1, c0:c1]

    # 3) Fallback to min_h/min_w if given
    if (min_h and min_h > 0) and (min_w and min_w > 0):
        return tokens_grid[0:min(MAX_GRID_SIZE, min_h), 0:min(MAX_GRID_SIZE, min_w)]

    raise ValueError("depad_tokens_to_content: No EOS and no content bbox; cannot crop.")


# --- Inference ---

@torch.no_grad()
def run_inference_on_test_inputs(model: nn.Module, task_id: str, json_path: str, device: torch.device) -> list[list[list[list[int]]]]:
    """For each test grid: run inference over static augs, then DEPAD (crop), invert transforms, and vote; returns top-2."""    
    with open(json_path, 'r') as f:
        all_tasks = json.load(f)
    task_data = all_tasks[task_id]
    test_items = task_data.get('test', [])

    static_augs = _get_or_create_static_augmentations(task_id, task_data, NUM_AUGMENTATIONS)
    results: list[list[list[list[int]]]] = []
    model.eval()

    if not any((tid == 0 and np.array_equal(cm, IDENTITY_CM)) for tid, cm in static_augs):
        static_augs = [(0, IDENTITY_CM)] + static_augs

    for item in test_items:
        inp = np.array(item['input'], dtype=np.int64)
        h, w = inp.shape
        aug_batches: list[torch.Tensor] = []
        aug_meta: list[tuple[int, np.ndarray, int, int]] = []
        # Build augmented inputs
        for trans_id, cm in static_augs:
            aug_in = dihedral_transform(cm[inp], trans_id)
            ah, aw = aug_in.shape
            toks = format_grid_with_padding(aug_in, pad_h=0, pad_w=0).unsqueeze(0)
            aug_batches.append(toks)
            aug_meta.append((trans_id, cm, ah, aw))
        if len(aug_batches) == 0:
            aug_batches.append(format_grid_with_padding(inp, pad_h=0, pad_w=0).unsqueeze(0))
            aug_meta.append((0, np.arange(10), h, w))
        all_toks = torch.cat(aug_batches, dim=0).to(device)

        # Run in chunks to save memory
        preds: list[np.ndarray] = []
        bs = 64
        for i in range(0, all_toks.size(0), bs):
            logits = model(all_toks[i:i+bs])
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)

        # Depad → Invert transforms → Vote
        counts: dict[tuple, int] = {}
        grids_cache: dict[tuple, list[list[int]]] = {}
        for k, pred_tokens in enumerate(preds):
            trans_id, cm, ah, aw = aug_meta[k]
            inv_tid = DIHEDRAL_INV_MAP[trans_id]
            inv_cm = inverse_color_map_arr(cm)

            # Reshape to 30x30 token grid
            tok_grid = pred_tokens.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
            # 1) DEPAD (crop) using EOS lines or color bbox (tokens domain)
            cropped_tokens = depad_tokens_to_content(tok_grid, min_h=ah, min_w=aw)
            # 2) Convert tokens→colors only inside cropped region (PAD/EOS -> 0 placeholder)
            colors_region = np.where(cropped_tokens >= 2, cropped_tokens - COLOR_OFFSET, 0).astype(np.int64)
            # 3) Invert color map then dihedral (return to original orientation)
            colors_region = inv_cm[colors_region]
            final = dihedral_transform(colors_region, inv_tid)

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
    # If the input is a flattened tensor, reshape it to a 2D grid
    if isinstance(input_grid, torch.Tensor) and input_grid.ndim == 1:
        input_grid = input_grid.view(MAX_GRID_SIZE, MAX_GRID_SIZE)
    if isinstance(output_grid, torch.Tensor) and output_grid.ndim == 1:
        output_grid = output_grid.view(MAX_GRID_SIZE, MAX_GRID_SIZE)

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
    cbar.set_label("Vocabulary (0:PAD, 1:EOS, 2-11:Colors)")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def visualize_augmented_dataset(task_id: str, json_path: str, num_augmentations: int = NUM_AUGMENTATIONS):
    """
    Interactive visualization of the entire augmented dataset with scrollable navigation.
    Shows both raw augmented data and formatted data side by side.
    """
    import matplotlib.widgets as widgets
    
    # Build the full dataset
    dataset = load_dataset_for_task(task_id, json_path, num_augmentations)
    total_examples = len(dataset.augmented_pairs)
    

    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Task {task_id}: Augmented Dataset Visualization", fontsize=16)
    
    # Current index
    current_idx = [0]
    
    # Colormap setup
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
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, VOCAB_SIZE, 1), cmap.N)
    
    def update_visualization(idx):
        """Update the visualization for the given index"""
        if idx < 0 or idx >= total_examples:
            return
            
        # Get the formatted data
        formatted_input, formatted_output = dataset[idx]
        
        # Convert to numpy for visualization
        if isinstance(formatted_input, torch.Tensor):
            formatted_input = formatted_input.cpu().numpy()
        if isinstance(formatted_output, torch.Tensor):
            formatted_output = formatted_output.cpu().numpy()
        
        # Reshape to 30x30 grids
        formatted_input = formatted_input.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
        formatted_output = formatted_output.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
        
        # Clear all subplots
        for ax in axes.flat:
            ax.clear()
        
        # Plot 1: Raw Input (extract original by removing padding)
        # Try to extract the original input by removing padding
        raw_input = depad_tokens_to_content(formatted_input)
        raw_input = np.where(raw_input >= 2, raw_input - COLOR_OFFSET, 0)
        
        axes[0, 0].imshow(raw_input, cmap=cmap, norm=norm, interpolation='nearest')
        axes[0, 0].set_title(f"Raw Input {idx+1}/{total_examples} ({raw_input.shape[0]}x{raw_input.shape[1]})")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        
        # Plot 2: Raw Output (extract original by removing padding)
        raw_output = depad_tokens_to_content(formatted_output)
        raw_output = np.where(raw_output >= 2, raw_output - COLOR_OFFSET, 0)
        
        axes[0, 1].imshow(raw_output, cmap=cmap, norm=norm, interpolation='nearest')
        axes[0, 1].set_title(f"Raw Output {idx+1}/{total_examples} ({raw_output.shape[0]}x{raw_output.shape[1]})")
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticklabels([])
        
        # Plot 3: Formatted Input
        axes[1, 0].imshow(formatted_input, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1, 0].set_title(f"Formatted Input (30x30)")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        
        # Plot 4: Formatted Output
        im = axes[1, 1].imshow(formatted_output, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1, 1].set_title(f"Formatted Output (30x30)")
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticklabels([])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=np.arange(VOCAB_SIZE), 
                           orientation='horizontal', fraction=0.1, pad=0.1)
        cbar.ax.set_xticklabels([f'{i}' for i in range(VOCAB_SIZE)])
        cbar.set_label("Vocabulary (0:PAD, 1:EOS, 2-11:Colors)")
        
        plt.tight_layout()
        fig.canvas.draw()
    
    def next_example(event):
        """Go to next example"""
        current_idx[0] = min(current_idx[0] + 1, total_examples - 1)
        update_visualization(current_idx[0])
    
    def prev_example(event):
        """Go to previous example"""
        current_idx[0] = max(current_idx[0] - 1, 0)
        update_visualization(current_idx[0])
    
    def on_key(event):
        """Handle keyboard navigation"""
        if event.key == 'right':
            next_example(None)
        elif event.key == 'left':
            prev_example(None)
        elif event.key == 'q':
            plt.close()
    
    # Create navigation buttons
    ax_prev = plt.axes([0.2, 0.02, 0.1, 0.04])
    ax_next = plt.axes([0.7, 0.02, 0.1, 0.04])
    btn_prev = widgets.Button(ax_prev, 'Previous')
    btn_next = widgets.Button(ax_next, 'Next')
    
    btn_prev.on_clicked(prev_example)
    btn_next.on_clicked(next_example)
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show first example
    update_visualization(0)
    
    plt.show()


if __name__ == '__main__':

    if Visualize:
        visualize_augmented_dataset(Task_id, file_path, NUM_AUGMENTATIONS)
    if Train:
        # Minimal training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")

        dataset = load_dataset_for_task(Task_id, file_path, num_augmentations=NUM_AUGMENTATIONS)
        # Turn off shuffling to ensure each batch contains the same transform applied to all training pairs
        # Batch size should be manually set to num_training_pairs for optimal grouping
        dataloader = DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=0)

        model = RecurrentTransformerNA(d_model=512, steps=RNN_iterations, nheads=16, nlayers=8).to(device)
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
            if len(test_items) > 0 and len(predictions) > 0:
                last_input_np = np.array(test_items[-1]['input'], dtype=np.int64)
                top2 = predictions[-1]
                for i, pred_grid in enumerate(top2[:2]):
                    pred_np = np.array(pred_grid, dtype=np.int64)
                    viz_input = last_input_np + COLOR_OFFSET
                    viz_output = pred_np + COLOR_OFFSET
                    visualize_datapoint(viz_input, viz_output, title=f"Task {Task_id}: Test Prediction Top-{i+1}")
            else:
                if len(test_items) == 0:
                    print("No test items found in challenges file; predictions file will be empty.")
                if len(predictions) == 0 and len(test_items) > 0:
                    print("Model produced no predictions for available test items.")
        except Exception as e:
            print(f"Visualization failed: {e}")

        # Visualize ALL generated test outputs grouped by duplicates and ranked by count
        try:
            with open(file_path, 'r') as f:
                all_tasks = json.load(f)
            test_items = all_tasks[Task_id].get('test', [])
            if len(test_items) > 0 and len(predictions) > 0:

                last_input_np = np.array(test_items[-1]['input'], dtype=np.int64)
                
                # Re-run inference for the last test item to get all individual predictions
                inp = last_input_np
                h, w = inp.shape
                aug_batches: list[torch.Tensor] = []
                aug_meta: list[tuple[int, np.ndarray, int, int]] = []
                
                # Build augmented inputs (same as in run_inference_on_test_inputs)
                static_augs = _get_or_create_static_augmentations(Task_id, all_tasks[Task_id], NUM_AUGMENTATIONS)
                for trans_id, cm in static_augs:
                    aug_in = dihedral_transform(cm[inp], trans_id)
                    ah, aw = aug_in.shape
                    toks = format_grid_with_padding(aug_in, pad_h=0, pad_w=0).unsqueeze(0)
                    aug_batches.append(toks)
                    aug_meta.append((trans_id, cm, ah, aw))
                if len(aug_batches) == 0:
                    aug_batches.append(format_grid_with_padding(inp, pad_h=0, pad_w=0).unsqueeze(0))
                    aug_meta.append((0, np.arange(10), h, w))
                all_toks = torch.cat(aug_batches, dim=0).to(device)

                # Run model inference
                preds: list[np.ndarray] = []
                bs = 64
                for i in range(0, all_toks.size(0), bs):
                    logits = model(all_toks[i:i+bs])
                    pred = logits.argmax(dim=-1).detach().cpu().numpy()
                    preds.append(pred)
                preds = np.concatenate(preds, axis=0)
                
                # Collect all individual predictions
                all_preds_with_counts = []
                for k, pred_tokens in enumerate(preds):
                    trans_id, cm, ah, aw = aug_meta[k]
                    inv_tid = DIHEDRAL_INV_MAP[trans_id]
                    inv_cm = inverse_color_map_arr(cm)

                    tok_grid = pred_tokens.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
                    cropped_tokens = depad_tokens_to_content(tok_grid, min_h=ah, min_w=aw)
                    colors_region = np.where(cropped_tokens >= 2, cropped_tokens - COLOR_OFFSET, 0).astype(np.int64)
                    colors_region = inv_cm[colors_region]
                    final = dihedral_transform(colors_region, inv_tid)
                    
                    all_preds_with_counts.append((final, trans_id, cm))

                # Group by duplicates and count
                counts: dict[bytes, tuple[np.ndarray, int, list[tuple[int, np.ndarray]]]] = {}
                for pred_grid, trans_id, cm in all_preds_with_counts:
                    key = pred_grid.tobytes()
                    if key in counts:
                        existing_grid, cnt, meta_list = counts[key]
                        counts[key] = (existing_grid, cnt + 1, meta_list + [(trans_id, cm)])
                    else:
                        counts[key] = (pred_grid, 1, [(trans_id, cm)])

                # Rank by count (descending)
                ranked = sorted(counts.values(), key=lambda x: x[1], reverse=True)
                total_votes = len(all_preds_with_counts)
                

                for rank, (grid, count, meta_list) in enumerate(ranked, 1):
                    print(f"  #{rank}: count={count}/{total_votes} ({(count/total_votes)*100:.1f}%)")
                    viz_input = last_input_np + COLOR_OFFSET
                    viz_output = grid + COLOR_OFFSET
                    visualize_datapoint(viz_input, viz_output, 
                                    title=f"Task {Task_id}: Test Output #{rank} (count={count}/{total_votes})")
            else:
                if len(test_items) == 0:
                    print("No test items found in challenges file; predictions file will be empty.")
                if len(predictions) == 0 and len(test_items) > 0:
                    print("Model produced no predictions for available test items.")
        except Exception as e:
            print(f"Visualization failed: {e}")

        # Visualize original training inputs and their model outputs (no augmentation, no translation)
        try:

            train_pairs = all_tasks[Task_id].get('train', [])
            
            for i, pair in enumerate(train_pairs):

                input_grid = np.array(pair['input'], dtype=np.int64)
                expected_output = np.array(pair['output'], dtype=np.int64)
                
                # Format input for model: no augmentation, no translation, just pad to 30x30 top-left
                # Create 30x30 grid with input in top-left corner, no EOS markers
                padded_input = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int64)
                h, w = input_grid.shape
                padded_input[:h, :w] = input_grid + COLOR_OFFSET
                model_input = torch.from_numpy(padded_input.flatten()).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    logits = model(model_input)
                    pred_tokens = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy()
                
                # Process output
                # Reshape to 30x30 and depad
                tok_grid = pred_tokens.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
                cropped_tokens = depad_tokens_to_content(tok_grid, min_h=h, min_w=w)
                # Convert tokens to colors
                pred_output = np.where(cropped_tokens >= 2, cropped_tokens - COLOR_OFFSET, 0).astype(np.int64)
                
                # Visualize: input vs expected vs predicted
                viz_input = input_grid + COLOR_OFFSET
                viz_expected = expected_output + COLOR_OFFSET
                viz_predicted = pred_output + COLOR_OFFSET
                
                # Use existing visualize_datapoint for input vs expected
                visualize_datapoint(viz_input, viz_expected, 
                                title=f"Task {Task_id}: Training Pair {i+1} - Input vs Expected")
                
                # Use existing visualize_datapoint for input vs predicted
                visualize_datapoint(viz_input, viz_predicted, 
                                title=f"Task {Task_id}: Training Pair {i+1} - Input vs Predicted")
                
        except Exception as e:
            print(f"Training visualization failed: {e}")