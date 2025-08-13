# =================================================================================================================
#  GPU PARALLEL SUPER KA-NCA
# =================================================================================================================
# This script parallelizes the execution of `super-ka-nca.py` by processing different groups of tasks
# concurrently on available hardware (multiple GPUs, MPS, or CPU cores).
#
# It follows the parallelization pattern from `gpu-ka-nca.py`, where a worker process is defined to handle
# one unit of work—in this case, a group of tasks that share the same number of I/O examples.
#
# HOW TO USE:
# 1. Scroll down to the `if __name__ == "__main__":` block.
# 2. Find the "--- PLATFORM CONFIGURATION ---" section.
# 3. Uncomment the block for the environment you are using (KAGGLE, GOOGLE COLAB, or LOCAL).
# 4. Ensure all other platform blocks are commented out.
# 5. Run the script.
# =================================================================================================================

import os
import json
import time
import torch
import sys
import datetime
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, List

# =================================================================================================================
# | SECTION 1: WORKER CODE                                                                                        |
# | This entire block of code will be written to a file at runtime. It is fully self-contained.                   |
# =================================================================================================================

WORKER_CODE = """
import os
import json
import time
import random
import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Utility functions (grid <-> tensor conversions) ---

def create_array_from_grid(grid: List[List[int]], grid_size: int, channels: int, n_classes: int) -> np.ndarray:
    \"\"\"Encode an ARC grid into a C-dim numpy array with one-hot colour channels.\"\"\"
    arr = np.zeros((grid_size, grid_size, channels), dtype=np.float32)
    arr[:, :, 0] = 1.0  # channel 0 = empty

    g = np.array(grid, dtype=np.int32)
    rows, cols = g.shape
    
    # Vectorized one-hot encoding for performance
    valid_mask = (g >= 0) & (g <= n_classes - 2)

    r_valid, c_valid = np.where(valid_mask)
    colours_valid = g[r_valid, c_valid]
    
    arr[r_valid, c_valid, :n_classes] = 0.0
    arr[r_valid, c_valid, colours_valid + 1] = 1.0
    return arr

def depad_grid(grid: List[List[int]], pad_val: int = -1) -> List[List[int]]:
    \"\"\"Crop surrounding pad_val rows/cols.\"\"\"
    if not grid or not grid[0]:
        return [[pad_val]]
    rows, cols = len(grid), len(grid[0])
    min_r, max_r = rows, -1
    min_c, max_c = cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != pad_val:
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)
    if max_r == -1:
        return [[pad_val]]
    return [row[min_c:max_c + 1] for row in grid[min_r:max_r + 1]]

def tensor_to_grid(state: torch.Tensor, n_classes: int, channels_per_task: int) -> List[List[int]]:
    \"\"\"Decode output grid (from output channel slice) back to int grid.\"\"\"
    out_slice = state[channels_per_task: channels_per_task + n_classes]
    pred = out_slice.argmax(dim=0).cpu().numpy()
    grid = (pred - 1).tolist()
    return grid

# --- Multi-task Cellular NCA model (grouped convolutions) ---

class MultiTaskCellularNN(nn.Module):
    def __init__(self, channels_per_task: int, hidden_per_task: int, num_tasks: int):
        super().__init__()
        self.T = num_tasks
        self.C0 = channels_per_task
        self.H0 = hidden_per_task

        perc_per_task = self.C0 * 9
        in_channels = perc_per_task * self.T
        out_hidden = self.H0 * self.T

        self.fc1 = nn.Conv2d(in_channels, out_hidden, kernel_size=1, groups=self.T)
        self.fc2 = nn.Conv2d(out_hidden, self.C0 * self.T, kernel_size=1, groups=self.T)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _get_neighbor_states(x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'cuda':
            B, C, H, W = x.shape
            patches = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H, W)
            patches = patches.permute(0, 2, 1, 3, 4)
            neighbors = torch.cat([patches[:, :4], patches[:, 5:]], dim=1)
            return neighbors.reshape(B, 8 * C, H, W)
        else:
            padded = F.pad(x, (1, 1, 1, 1))
            h, w = x.shape[2], x.shape[3]
            neigh = []
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dr, dc in offsets:
                neigh.append(padded[:, :, 1 + dr:1 + dr + h, 1 + dc:1 + dc + w])
            return torch.cat(neigh, 1)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        neighbors = self._get_neighbor_states(x)
        percep = torch.cat([x, neighbors], dim=1)
        B, _, H, W = percep.shape
        percep = percep.view(B, 9, self.T, self.C0, H, W)
        percep = percep.permute(0, 2, 1, 3, 4, 5)
        percep = percep.contiguous().view(B, self.T * 9 * self.C0, H, W)
        return percep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self.perceive(x)))
        return x + self.fc2(h)

# --- N-step wrapper (TorchScript-able) ---

class NCAStepper(nn.Module):
    def __init__(self, nca: nn.Module):
        super().__init__()
        self.nca = nca

    def forward(self, state: torch.Tensor, steps: int, fire_rate: float):
        if fire_rate < 1.0:
            mask = torch.empty_like(state[:, :1])
            for _ in range(steps):
                dx = self.nca(state) - state
                mask.bernoulli_(fire_rate)
                state = state + dx * mask
            return state
        else:
            for _ in range(steps):
                state = self.nca(state)
            return state

    @torch.jit.export
    def run(self, state: torch.Tensor, steps: int, fire_rate: float):
        return self.forward(state, steps, fire_rate)

# --- Training & inference helpers for a group of tasks ---

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def build_state_tensor(tasks_inputs: List[np.ndarray]) -> torch.Tensor:
    arr_concat = np.concatenate(tasks_inputs, axis=2)
    tensor = torch.tensor(arr_concat).permute(2, 0, 1).unsqueeze(0)
    return tensor

def slice_state_tensor(state: torch.Tensor, idx: int, channels_per_task: int) -> torch.Tensor:
    start = idx * channels_per_task
    end = start + channels_per_task
    return state[:, start:end]

def train_and_predict_for_group(group_task_ids: List[str],
                                preprocessed_group_data: Dict[str, Any],
                                device: torch.device,
                                hparams: Dict[str, Any]) -> Dict[str, List[List[int]]]:
    T = len(group_task_ids)
    if T == 0:
        return {}
    
    C0 = hparams['channels_per_task']
    grid_size = hparams['grid_size']
    n_classes = hparams['n_classes']
    num_train_pairs_per_task = [len(preprocessed_group_data[tid].get('train', [])) for tid in group_task_ids]

    model = MultiTaskCellularNN(C0, hparams['hidden_per_task'], T).to(device)
    stepper = torch.jit.script(NCAStepper(model)) if device.type == 'cuda' else NCAStepper(model)
    optimiser = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
        
    num_test_pairs_per_task = [len(preprocessed_group_data[tid].get('test', [])) for tid in group_task_ids]
    P = num_train_pairs_per_task[0] + num_test_pairs_per_task[0]

    inp_batch = torch.zeros(P, T * C0, grid_size, grid_size, device=device)
    tgt_batch = torch.zeros(P, T * C0, grid_size, grid_size, device=device)
    loss_mask = torch.zeros_like(tgt_batch)
    C0_half = C0 // 2

    for task_idx, tid in enumerate(group_task_ids):
        task_data = preprocessed_group_data[tid]
        start_ch = task_idx * C0
        end_ch = start_ch + C0

        # Process training pairs from pre-cached arrays
        for pair_idx, pair in enumerate(task_data['train']):
            input_arr_15ch = pair['input']
            output_arr_15ch = pair['output']
            zeros_arr_15ch = np.zeros_like(input_arr_15ch)
            input_arr_30ch = np.concatenate((input_arr_15ch, zeros_arr_15ch), axis=2)
            target_arr_30ch = np.concatenate((input_arr_15ch, output_arr_15ch), axis=2)
            inp_batch[pair_idx, start_ch:end_ch] = torch.from_numpy(input_arr_30ch).permute(2,0,1)
            tgt_batch[pair_idx, start_ch:end_ch] = torch.from_numpy(target_arr_30ch).permute(2,0,1)

        # Process test pairs from pre-cached arrays (input is used for both input and target)
        num_train = num_train_pairs_per_task[task_idx]
        for pair_idx, pair in enumerate(task_data['test']):
            batch_idx = num_train + pair_idx
            input_arr_15ch = pair['input']
            zeros_arr_15ch = np.zeros_like(input_arr_15ch)
            input_arr_30ch = np.concatenate((input_arr_15ch, zeros_arr_15ch), axis=2)
            target_arr_30ch = np.concatenate((input_arr_15ch, input_arr_15ch), axis=2)
            inp_batch[batch_idx, start_ch:end_ch] = torch.from_numpy(input_arr_30ch).permute(2,0,1)
            tgt_batch[batch_idx, start_ch:end_ch] = torch.from_numpy(target_arr_30ch).permute(2,0,1)

    for task_idx, num_train in enumerate(num_train_pairs_per_task):
        start_ch_task = task_idx * C0
        loss_mask[:num_train, start_ch_task : start_ch_task + C0, :, :] = 1.0
        loss_mask[num_train:, start_ch_task : start_ch_task + C0_half, :, :] = 1.0

    model.train()
    for it in range(hparams['num_iterations']):
        optimiser.zero_grad(set_to_none=True)
        state = inp_batch.clone()
        state = stepper.run(state, int(hparams['train_steps']), float(hparams['fire_rate']))
        loss = ((state - tgt_batch)**2 * loss_mask).sum() / loss_mask.sum()
        
        coef = hparams.get('weight_variance_coef', 0.0)
        if coef > 0:
            w1_chunks = torch.split(model.fc1.weight, model.H0, dim=0)
            b1_chunks = torch.split(model.fc1.bias, model.H0, dim=0)
            w2_chunks = torch.split(model.fc2.weight, model.C0, dim=0)
            b2_chunks = torch.split(model.fc2.bias, model.C0, dim=0)
            total_variance = sum(
                w1.var(unbiased=False) + b1.var(unbiased=False) + w2.var(unbiased=False) + b2.var(unbiased=False)
                for w1, b1, w2, b2 in zip(w1_chunks, b1_chunks, w2_chunks, b2_chunks)
            )
            loss += coef * total_variance

        loss.backward()
        max_norm = hparams['max_norm']
        if max_norm > 0 and model.fc1.weight.grad is not None:
            w1_grad_chunks = torch.split(model.fc1.weight.grad, model.H0, dim=0)
            b1_grad_chunks = torch.split(model.fc1.bias.grad, model.H0, dim=0)
            w2_grad_chunks = torch.split(model.fc2.weight.grad, model.C0, dim=0)
            b2_grad_chunks = torch.split(model.fc2.bias.grad, model.C0, dim=0)
            for t in range(model.T):
                torch.nn.utils.clip_grad_norm_([w1_grad_chunks[t], b1_grad_chunks[t], w2_grad_chunks[t], b2_grad_chunks[t]], max_norm)
        
        optimiser.step()

    model.eval()
    submission_fragment = {}
    with torch.no_grad():
        for task_idx, task_id in enumerate(group_task_ids):
            test_inputs = preprocessed_group_data[task_id]['test']
            predictions_for_task = []
            for test_case in test_inputs:
                input_arr = test_case['input'] # Already a preprocessed numpy array
                zeros_arr = np.zeros_like(input_arr)
                full_input = np.concatenate((input_arr, zeros_arr), axis=2)
                task_inputs_blank = [np.zeros_like(full_input) for _ in range(T)]
                task_inputs_blank[task_idx] = full_input
                inp_tensor_single = build_state_tensor(task_inputs_blank).to(device)
                state = stepper.run(inp_tensor_single, int(hparams['prediction_steps']), float(hparams['fire_rate']))
                slice_pred = slice_state_tensor(state, task_idx, C0).squeeze(0)
                grid = tensor_to_grid(slice_pred, n_classes, C0 // 2)
                depadded = depad_grid(grid)
                final_grid = [[max(0, c) for c in row] for row in depadded]
                predictions_for_task.append({"attempt_1": final_grid, "attempt_2": final_grid})
            submission_fragment[task_id] = predictions_for_task

    # Explicitly clean up to prevent memory leaks in long-running pool workers
    del model
    del stepper
    del optimiser
    del inp_batch
    del tgt_batch
    del loss_mask
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return submission_fragment

# --- Worker wrapper function for the pool ---
def worker_process(args):
    group_task_ids, preprocessed_group_data, device_str, hparams = args
    device = torch.device(device_str)
    
    if not group_task_ids:
        return {}

    try:
        set_seed(hparams['seed'])
        
        # Announce which group is starting on which device
        # Using flush=True is important in multiprocessing contexts to see logs immediately
        print(f"Starting group {group_task_ids} ({len(group_task_ids)} tasks) on device {device_str}", flush=True)
        
        submission_fragment = train_and_predict_for_group(
            group_task_ids, preprocessed_group_data, device, hparams
        )
        print(f"Finished group {group_task_ids} on device {device_str}", flush=True)
        return submission_fragment
    except Exception as e:
        print(f"!!! ERROR processing group {group_task_ids} on device {device_str}: {e}", flush=True)
        return {}
"""

# =================================================================================================================
# | SECTION 2: MAIN EXECUTION SCRIPT                                                                              |
# =================================================================================================================

def preprocess_and_cache_data(challenges: Dict[str, Any], hparams: Dict[str, Any], create_array_func) -> Dict[str, Any]:
    """
    Pre-converts all input/output grids into NumPy arrays to avoid redundant processing
    in parallel workers.
    """
    print("Preprocessing and caching all grid data... this may take a moment.")
    C0_half = hparams['channels_per_task'] // 2
    grid_size = hparams['grid_size']
    n_classes = hparams['n_classes']
    
    cached_data = defaultdict(lambda: {'train': [], 'test': []})
    
    # Use tqdm for a progress bar during this potentially long step
    for task_id, task_data in tqdm(challenges.items(), desc="Preprocessing"):
        # Cache training pairs
        for pair in task_data.get('train', []):
            input_arr = create_array_func(pair['input'], grid_size, C0_half, n_classes)
            output_arr = create_array_func(pair['output'], grid_size, C0_half, n_classes)
            cached_data[task_id]['train'].append({'input': input_arr, 'output': output_arr})
            
        # Cache test inputs
        for test_case in task_data.get('test', []):
            input_arr = create_array_func(test_case['input'], grid_size, C0_half, n_classes)
            cached_data[task_id]['test'].append({'input': input_arr})
            
    print("Finished preprocessing.")
    return dict(cached_data)


if __name__ == "__main__":
    script_start_time = time.time()

    # --- PLATFORM CONFIGURATION: UNCOMMENT THE BLOCK FOR YOUR CURRENT ENVIRONMENT ---

    # # --- 1. KAGGLE CONFIG ---
    # PLATFORM_NAME = "Kaggle"
    # ARC_DATA_DIR = "/kaggle/input/arc-prize-2024"
    # INPUT_JSON_FILENAME = "challenges.json"
    # OUTPUT_DIR = "/kaggle/working/"
    # WORKERS_PER_GPU = 1
    # LOCAL_WORKERS = 4
    # MAX_GROUP_SIZE = 20 # Split groups larger than this for better load balancing

    # --- 2. GOOGLE COLAB CONFIG ---
    PLATFORM_NAME = "Google Colab"
    ARC_DATA_DIR = "/content"
    INPUT_JSON_FILENAME = "challenges.json"
    OUTPUT_DIR = os.path.join("/content", f"gpu_super_ka_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}")
    WORKERS_PER_GPU = 6
    LOCAL_WORKERS = 6
    MAX_GROUP_SIZE = 20 # Split groups larger than this for better load balancing

    # # --- 3. LOCAL MAC/PC CONFIG ---
    # PLATFORM_NAME = "Local Mac/PC"
    # ARC_DATA_DIR = "../../dataset/script-tests/grouped-tasks-0-4x"
    # INPUT_JSON_FILENAME = "challenges.json"
    # OUTPUT_DIR = os.path.join("../runs", f"gpu_super_ka_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}")
    # WORKERS_PER_GPU = 5 # Adjust based on your GPU memory
    # LOCAL_WORKERS = 5   # Adjust based on your CPU cores
    # MAX_GROUP_SIZE = 10 # Split groups larger than this for better load balancing
    # # ------------------------------------------------------------------------------------

    print(f"Running on platform: {PLATFORM_NAME}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    WORKER_FILENAME = os.path.join(OUTPUT_DIR, "temp_super_nca_worker.py")

    # Hyperparameters from super-ka-nca.py
    HPARAMS: Dict[str, Any] = {
        "grid_size": 30,
        "n_classes": 11,
        "channels_per_task": 30,
        "hidden_per_task": 16,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_iterations": 3000,
        "train_steps": 30,
        "prediction_steps": 30,
        "max_norm": 1.0,
        "fire_rate": 1.0,
        "seed": 42,
        "weight_variance_coef": 1e-3,
    }

    # --- DYNAMIC WORKER FILE CREATION ---
    try:
        with open(WORKER_FILENAME, "w") as f:
            f.write(WORKER_CODE)
        sys.path.insert(0, OUTPUT_DIR)
        from temp_super_nca_worker import worker_process, create_array_from_grid
        print(f"Successfully wrote and imported worker from {WORKER_FILENAME}")
    except Exception as e:
        print(f"FATAL: Could not write or import the worker file: {e}")
        exit()

    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
             mp.set_start_method('spawn', force=True)
             print("Multiprocessing start method set to 'spawn'.")
    except (RuntimeError, AttributeError): pass

    # --- DYNAMIC DEVICE AND WORKER SCALING ---
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        N_WORKERS = num_gpus * WORKERS_PER_GPU
        print(f"Found {num_gpus} CUDA GPU(s). Using {N_WORKERS} workers ({WORKERS_PER_GPU} per GPU).")
    elif torch.backends.mps.is_available():
        devices = ["mps"]
        N_WORKERS = LOCAL_WORKERS
        print(f"Found Apple MPS. Using 'mps' device with {N_WORKERS} workers.")
    else:
        devices = ["cpu"]
        N_WORKERS = LOCAL_WORKERS
        print(f"No GPU found. Using 'cpu' device with {N_WORKERS} workers.")
    print(f"Groups will be distributed across: {devices}")

    # --- DATA LOADING AND GROUP PREPARATION ---
    INPUT_JSON_FILE = os.path.join(ARC_DATA_DIR, INPUT_JSON_FILENAME)
    try:
        with open(INPUT_JSON_FILE, 'r') as f:
            challenges = json.load(f)
        print(f"Loaded {len(challenges)} tasks from {INPUT_JSON_FILE}")

        # Preprocess and cache all grid data to be passed to workers
        preprocessed_data = preprocess_and_cache_data(challenges, HPARAMS, create_array_from_grid)
        # We no longer need the raw challenges dict, free up memory
        del challenges

    except FileNotFoundError:
        print(f"FATAL: Input file not found at {INPUT_JSON_FILE}")
        exit()
    
    # --- Group tasks by the total number of train+test pairs (from super-ka-nca.py) ---
    groups = defaultdict(list)
    for task_id, task_data in preprocessed_data.items():
        num_pairs = len(task_data['train']) + len(task_data['test'])
        if num_pairs > 0:
            groups[num_pairs].append(task_id)

    print(f"Created {len(groups)} groups based on I/O pair counts.")
    
    # --- SPLIT LARGE GROUPS FOR BETTER LOAD BALANCING ---
    all_groups_to_process = []
    for pair_count, group_ids in sorted(groups.items()):
        if len(group_ids) > MAX_GROUP_SIZE:
            print(f"Group with {pair_count} pairs ({len(group_ids)} tasks) is larger than MAX_GROUP_SIZE ({MAX_GROUP_SIZE}). Splitting.")
            for i in range(0, len(group_ids), MAX_GROUP_SIZE):
                all_groups_to_process.append(group_ids[i:i + MAX_GROUP_SIZE])
        else:
            all_groups_to_process.append(group_ids)

    print(f"Total work items after splitting: {len(all_groups_to_process)}")

    group_args = []
    for i, group_ids in enumerate(all_groups_to_process):
        device_str = devices[i % len(devices)]
        # Pass only the required subset of preprocessed data to each worker
        group_data = {tid: preprocessed_data[tid] for tid in group_ids}
        group_args.append((
            group_ids, group_data, device_str, HPARAMS
        ))

    # --- PARALLEL PROCESSING ---
    final_submission = {}
    print(f"\nStarting processing for {len(group_args)} groups with {N_WORKERS} workers...")
    
    # Set a timeout for the pool (e.g., 1 hour per group) to prevent hangs
    timeout_seconds = 3600 * len(group_args) 

    with mp.Pool(processes=N_WORKERS) as pool:
        # Use imap_unordered for better progress visibility and resource utilization
        results_iterator = pool.imap_unordered(worker_process, group_args)
        
        for submission_fragment in tqdm(results_iterator, total=len(group_args)):
            if submission_fragment:
                final_submission.update(submission_fragment)

    # --- PROCESS AND SAVE RESULTS ---
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(final_submission, f, indent=4)

    # Clean up the temporary worker file
    try:
        os.remove(WORKER_FILENAME)
        print(f"Cleaned up temporary worker file: {WORKER_FILENAME}")
    except OSError as e:
        print(f"Could not remove temporary worker file: {e}")

    total_time = time.time() - script_start_time
    print("\n-----------------------------------------")
    print(f"Success! Submission file saved to {SUBMISSION_FILE}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print("-----------------------------------------")
