# Modification of vanilla-v5
# Instead of predicting outputs given inputs,
# This predicts both inputs and outputs given a seed

# NCA^n(seed) = input
# NCA^n(input) = output

# Loss =  MSE(NCA^n(seed) - actual input) (both test and train input grids) + MSE(NCA^n(input) - actual output) (only train grids)

# Note that we do NOT want to leak the actual test output, so the second term only trains on inputs
# Note 2: For ith pair, the seed (starting grid) is just all 0 input with a '1' pixel at row i

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import sys
import subprocess

import datetime

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")  # Format: YYMMDD_HHMMSS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from evaluate import visualise_training_progression
except ImportError:
    visualise_training_progression = None

# --- 1. SETTINGS & PATHS ---
# Uncomment for local mac silicon run
ARC_DATA_DIR = "../../dataset/script-tests/grouped-tasks-0dfd9992"
OUTPUT_DIR = os.path.join("../runs", f"test_{timestamp}")
VISUALISE = True  # Set to True to generate visualization.pdf at the end of execution. Only for single-threaded NCA.
EVALUATE_SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "evaluate.py")
)

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

# Hyperparameters
HPARAMS: Dict[str, Any] = {
    "grid_size": 21,
    "n_classes": 11,
    "in_channels": 20,  # 11 for color one-hot, 9 for hidden state
    "hidden_channels": 9,
    "nn_hidden_dim": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_iterations": 2000,
    "seed_to_input_steps": 1,
    "input_to_output_steps": 30,
    "max_norm": 1.0,
    "checkpoint_interval": 200,
    "inference_start_mode": "both",  # Options: from_input, from_seed, both
}


def resolve_inference_modes(mode: str) -> List[str]:
    """Return the concrete inference rollouts that should be executed."""
    valid_modes = {"from_input", "from_seed", "both"}
    if mode not in valid_modes:
        raise ValueError(
            f"inference_start_mode must be one of {sorted(valid_modes)}, got '{mode}'."
        )
    if mode == "both":
        return ["from_input", "from_seed"]
    return [mode]


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
        padded_state = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
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

                neighbor_slice = padded_state[
                    :, :, start_row:end_row, start_col:end_col
                ]
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


def create_array_from_grid(
    small_grid: List[List[int]], grid_size: int, in_channels: int, n_classes: int
) -> np.ndarray:
    """Creates a (grid_size, grid_size, in_channels) numpy array from a small grid."""
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    # Channel 0 is the "empty" channel
    arr[:, :, 0] = 1.0

    small_grid_np = np.array(small_grid, dtype=np.int32)
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


def create_seed_tensor(
    example_idx: int, grid_args: Tuple[int, int, int]
) -> torch.Tensor:
    """Creates a seed tensor that has a single '1' pixel on the row matching example_idx."""
    grid_size = grid_args[0]
    if grid_size <= 0:
        raise ValueError("grid_size must be positive to create a seed tensor.")

    seed_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    row_idx = min(example_idx, grid_size - 1)
    seed_grid[row_idx][0] = 1
    seed_array = create_array_from_grid(seed_grid, *grid_args)
    return torch.tensor(seed_array).permute(2, 0, 1)


def mse_on_visible_channels(
    prediction: torch.Tensor, target: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Compute the MSE using only visible color channels, leaving hidden state unconstrained."""
    return F.mse_loss(prediction[:, :n_classes, :, :], target[:, :n_classes, :, :])


# --- 4. The "Workhorse" Function ---


def train_and_predict_for_task(
    task_id: str,
    train_pairs: List[Dict],
    test_inputs: List[Dict],
    device: torch.device,
    hparams: Dict[str, Any],
    inference_modes: List[str] = None,
) -> Tuple[
    Dict[str, List[List[int]]], Dict[str, Dict[int, List[Dict[str, List[List[int]]]]]]
]:
    """
    Initializes, trains, and uses an NCA model for a single task.
    Returns:
        predictions_by_mode: dict mapping inference rollout type -> predicted grids.
        checkpoint_predictions_by_mode: dict mapping inference rollout type -> {iteration: formatted predictions}
    """
    if inference_modes is None:
        inference_modes = ["from_input"]
    checkpoint_interval = hparams.get("checkpoint_interval")
    checkpoint_predictions_by_mode: Dict[
        str, Dict[int, List[Dict[str, List[List[int]]]]]
    ] = {mode: {} for mode in inference_modes}
    # 1. Model & Optimizer Initialization
    model = CellularNN(
        hparams["in_channels"], hparams["n_classes"], hparams["nn_hidden_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )

    # 2. Data Preparation (No DataLoader)
    grid_args = (hparams["grid_size"], hparams["in_channels"], hparams["n_classes"])
    train_input_tensors = [
        torch.tensor(create_array_from_grid(p["input"], *grid_args)).permute(2, 0, 1)
        for p in train_pairs
    ]
    train_target_tensors = [
        torch.tensor(create_array_from_grid(p["output"], *grid_args)).permute(2, 0, 1)
        for p in train_pairs
    ]
    test_input_tensors = [
        torch.tensor(create_array_from_grid(p["input"], *grid_args)).permute(2, 0, 1)
        for p in test_inputs
    ]
    all_input_tensors = train_input_tensors + test_input_tensors
    seed_tensors = [
        create_seed_tensor(idx, grid_args) for idx in range(len(all_input_tensors))
    ]

    train_input_batch = torch.stack(train_input_tensors).to(device)
    train_target_batch = torch.stack(train_target_tensors).to(device)
    all_input_batch = torch.stack(all_input_tensors).to(device)
    seed_batch = torch.stack(seed_tensors).to(device)

    def format_prediction(state_tensor: torch.Tensor) -> List[List[int]]:
        grid_from_tensor = tensor_to_grid(state_tensor.squeeze(0), hparams["n_classes"])
        depadded_grid = depad_grid(grid_from_tensor)
        return [[max(0, cell) for cell in row] for row in depadded_grid]

    def run_from_input() -> List[List[int]]:
        predictions = []
        steps = hparams["input_to_output_steps"]
        for test_case in test_inputs:
            test_input_grid = test_case["input"]
            inp_array = create_array_from_grid(test_input_grid, *grid_args)
            state = torch.tensor(inp_array).permute(2, 0, 1).unsqueeze(0).to(device)
            for _ in range(steps):
                state = model(state)
            predictions.append(format_prediction(state))
        return predictions

    def run_from_seed() -> List[List[int]]:
        predictions = []
        seed_offset = len(train_pairs)
        for idx, _ in enumerate(test_inputs):
            seed_tensor = create_seed_tensor(seed_offset + idx, grid_args).to(device)
            state = seed_tensor.unsqueeze(0)
            for _ in range(hparams["seed_to_input_steps"]):
                state = model(state)
            for _ in range(hparams["input_to_output_steps"]):
                state = model(state)
            predictions.append(format_prediction(state))
        return predictions

    def infer_for_modes(
        requested_modes: List[str] = None,
    ) -> Dict[str, List[List[int]]]:
        requested = requested_modes or inference_modes
        preds: Dict[str, List[List[int]]] = {}
        was_training = model.training
        if was_training:
            model.eval()
        with torch.no_grad():
            if "from_input" in requested:
                preds["from_input"] = run_from_input()
            if "from_seed" in requested:
                preds["from_seed"] = run_from_seed()
        if was_training:
            model.train()
        return preds

    # 3. Training Loop
    model.train()
    for i in range(hparams["num_iterations"]):
        optimizer.zero_grad()

        # NCA update steps
        seed_state = seed_batch
        for _ in range(hparams["seed_to_input_steps"]):
            seed_state = model(seed_state)

        seed_loss = mse_on_visible_channels(
            seed_state, all_input_batch, hparams["n_classes"]
        )

        train_state = seed_state[: len(train_pairs)].detach()
        for _ in range(hparams["input_to_output_steps"]):
            train_state = model(train_state)

        output_loss = mse_on_visible_channels(
            train_state, train_target_batch, hparams["n_classes"]
        )

        loss = seed_loss + output_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["max_norm"])
        optimizer.step()

        current_iter = i + 1
        if checkpoint_interval and current_iter % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"{task_id}_iter{current_iter:04d}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_preds = infer_for_modes()
            for mode in inference_modes:
                formatted = [
                    {"attempt_1": grid, "attempt_2": grid}
                    for grid in checkpoint_preds.get(mode, [])
                ]
                checkpoint_predictions_by_mode[mode][current_iter] = formatted
            print(
                f"  Saved checkpoint {checkpoint_path} (iter {current_iter}) "
                "and recorded inference outputs."
            )

        if current_iter % 200 == 0 or current_iter == 1:
            print(
                f"  Task {task_id}, Iter {current_iter:04d}: SeedLoss={seed_loss.item():.4f}, OutputLoss={output_loss.item():.4f}"
            )

    # 4. Save Final Checkpoint
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{task_id}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"  Saved final model for task {task_id} to {final_checkpoint_path}")

    predictions_by_mode = infer_for_modes()
    return predictions_by_mode, checkpoint_predictions_by_mode


# --- 5. Main Execution Block ---

if __name__ == "__main__":
    script_start_time = time.time()

    # Setup
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    inference_modes = resolve_inference_modes(HPARAMS["inference_start_mode"])
    print(f"Inference rollout(s): {', '.join(inference_modes)}")

    # Load data
    try:
        with open(INPUT_JSON_FILE, "r") as f:
            challenges = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Input JSON not found at {INPUT_JSON_FILE}")
        print("Please check the `ARC_DATA_DIR` and `INPUT_JSON_FILE` variables.")
        exit()
    solutions = None
    if VISUALISE:
        solutions_path = os.path.join(ARC_DATA_DIR, "solutions.json")
        try:
            with open(solutions_path, "r") as f:
                solutions = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: solutions.json not found at {solutions_path}.")
            print("         Training progression visualization will be skipped.")
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to parse solutions.json: {e}")
            print("         Training progression visualization will be skipped.")
            solutions = None

    submission_by_mode: Dict[str, Dict[str, List[Dict[str, List[List[int]]]]]] = {
        mode: {} for mode in inference_modes
    }
    checkpoint_history_by_mode: Dict[
        str, Dict[str, Dict[int, List[Dict[str, List[List[int]]]]]]
    ] = {mode: {} for mode in inference_modes}

    # Main processing loop
    task_ids = list(challenges.keys())
    print(f"Found {len(task_ids)} tasks in {INPUT_JSON_FILE}")

    for i, task_id in enumerate(task_ids):
        task_data = challenges[task_id]
        print(f"\n--- Processing task {i + 1}/{len(task_ids)}: {task_id} ---")

        train_pairs = task_data["train"]
        # The 'test' field in training files are pairs, but in test files are just inputs.
        # We handle this by only looking at the 'input' key.
        test_inputs = task_data["test"]

        if not train_pairs:
            print(f"  Skipping task {task_id}: No training pairs.")
            predictions_by_mode = {
                mode: [[[0]] for _ in test_inputs] for mode in inference_modes
            }
            checkpoint_preds_by_mode = {mode: {} for mode in inference_modes}
        else:
            # The magic happens here
            predictions_by_mode, checkpoint_preds_by_mode = train_and_predict_for_task(
                task_id,
                train_pairs,
                test_inputs,
                device,
                HPARAMS,
                inference_modes=inference_modes,
            )

        # Format for submission.json per mode
        for mode in inference_modes:
            if mode not in predictions_by_mode:
                raise RuntimeError(
                    f"Missing predictions for inference mode '{mode}' on task {task_id}"
                )
            formatted_predictions = [
                {"attempt_1": grid, "attempt_2": grid}
                for grid in predictions_by_mode[mode]
            ]
            submission_by_mode[mode][task_id] = formatted_predictions
            checkpoint_history_by_mode[mode][task_id] = checkpoint_preds_by_mode.get(
                mode, {}
            )

    # Save submission files (one per inference mode if needed)
    submission_paths: Dict[str, str] = {}
    mode_output_dirs: Dict[str, str] = {}
    if len(inference_modes) == 1:
        only_mode = inference_modes[0]
        submission_paths[only_mode] = SUBMISSION_FILE
        mode_output_dirs[only_mode] = OUTPUT_DIR
    else:
        for mode in inference_modes:
            mode_dir = os.path.join(OUTPUT_DIR, mode)
            submission_paths[mode] = os.path.join(mode_dir, "submission.json")
            mode_output_dirs[mode] = mode_dir

    for mode in inference_modes:
        os.makedirs(mode_output_dirs[mode], exist_ok=True)
        submission_path = submission_paths[mode]
        with open(submission_path, "w") as f:
            json.dump(submission_by_mode[mode], f)
        print(f"Saved '{mode}' submission to {submission_path}")

    if VISUALISE:
        dataset_path = os.path.abspath(ARC_DATA_DIR)

        if visualise_training_progression is None:
            print(
                "\nvisualise_training_progression import unavailable; skipping training progression visualization."
            )
        elif solutions is None:
            print(
                "\nsolutions.json missing or invalid; skipping training progression visualization."
            )
        else:
            print("\nGenerating training progression visualizations...")
            for mode in inference_modes:
                mode_history = checkpoint_history_by_mode.get(mode, {})
                has_history = any(
                    iteration_dict for iteration_dict in mode_history.values()
                )
                if not has_history:
                    print(
                        f"  -> No checkpoint predictions recorded for mode '{mode}', skipping progression plot."
                    )
                    continue
                try:
                    visualise_training_progression(
                        all_checkpoint_predictions=mode_history,
                        task_ids=task_ids,
                        challenges_dict=challenges,
                        solutions_dict=solutions,
                        output_dir_path=mode_output_dirs[mode],
                        final_submission_path=os.path.abspath(submission_paths[mode]),
                    )
                    print(
                        f"  -> Training progression PDF saved under {mode_output_dirs[mode]}"
                    )
                except Exception as e:
                    print(
                        f"  -> Error while generating training progression for mode '{mode}': {e}"
                    )

        print("\nStarting final submission visualizations...")
        for mode in inference_modes:
            submission_path = os.path.abspath(submission_paths[mode])
            print(f"  -> Visualizing mode '{mode}' using {submission_path}")
            try:
                cmd = [
                    sys.executable,
                    EVALUATE_SCRIPT_PATH,
                    "--submission_file",
                    submission_path,
                    "--dataset",
                    dataset_path,
                    "--visualize",
                ]
                print(f"     Executing: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                viz_path = os.path.join(
                    os.path.dirname(submission_path), "visualization.pdf"
                )
                print(f"     Visualization saved to {viz_path}")
            except Exception as e:
                print(f"     Error during visualization for mode '{mode}': {e}")

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    print("\n-----------------------------------------")
    if len(submission_paths) == 1:
        only_mode = inference_modes[0]
        print(f"Success! Submission file saved to {submission_paths[only_mode]}")
    else:
        print("Success! Submission files generated:")
        for mode in inference_modes:
            print(f"  [{mode}] {submission_paths[mode]}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("-----------------------------------------")
