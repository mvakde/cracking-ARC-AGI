# This is the evaluation script for the ARC-AGI challenge.
# Generates the result metrics and visualization.pdf files.

import json
import csv
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import numpy as np
import glob

# from google.colab import drive
# drive.mount('/content/drive')

# Helper for padding grids
def _pad_grid(grid, target_h, target_w, pad_value=-1):
    """Pads a grid to target dimensions with a specified pad_value."""
    padded = [[pad_value] * target_w for _ in range(target_h)]
    # Ensure grid is not empty and has inner lists before iterating
    if grid and grid[0] and isinstance(grid[0], list):
        for r_idx in range(len(grid)):
            for c_idx in range(len(grid[0])):
                # Copy existing values to the padded grid
                padded[r_idx][c_idx] = grid[r_idx][c_idx]
    return padded

def calculate_union_accuracy(pred_grid, true_grid, pad_value=-1):
    """
    Calculates pixel accuracy between two grids based on their union.
    Cells present in one grid but not the other (within the union bounding box)
    are treated as mismatches against the pad_value.
    Returns (matched_pixels, total_union_pixels).
    """
    true_h = len(true_grid)
    true_w = len(true_grid[0]) if true_h > 0 else 0
    pred_h = len(pred_grid)
    pred_w = len(pred_grid[0]) if pred_h > 0 else 0

    common_h = max(true_h, pred_h)
    common_w = max(true_w, pred_w)

    padded_true = _pad_grid(true_grid, common_h, common_w, pad_value)
    padded_pred = _pad_grid(pred_grid, common_h, common_w, pad_value)

    matched_pixels = 0
    total_union_pixels = 0

    # Iterate over the bounding box of the union
    for r in range(common_h):
        for c in range(common_w):
            # Check if the cell (r, c) exists in the original prediction or ground truth grid
            in_pred_orig = (r < pred_h and c < pred_w)
            in_true_orig = (r < true_h and c < true_w)

            if in_pred_orig or in_true_orig:
                total_union_pixels += 1 # This cell is part of the union

                val_pred = padded_pred[r][c]
                val_true = padded_true[r][c]

                # Only count as a match if their values are identical at this position
                # (This implicitly handles cases where one is padding and the other is not - they won't match)
                if val_pred == val_true:
                    matched_pixels += 1

    return matched_pixels, total_union_pixels

def visualise_training_progression(
    all_checkpoint_predictions: dict,
    task_ids: list,
    challenges_dict: dict,
    solutions_dict: dict,
    output_dir_path: str,
    final_submission_path: str
):
    """
    Generates a combined PDF visualizing the model's predictions over training iterations,
    including intermediate checkpoints and the final submission.
    """
    print("\nStarting combined visualization for all submission files...")
    
    intermediate_submissions_dir = os.path.join(output_dir_path, "intermediate_submissions")
    os.makedirs(intermediate_submissions_dir, exist_ok=True)
    
    submission_timeline = []
    if all_checkpoint_predictions:
        for task_idx, task_id in enumerate(task_ids):
            if task_id in all_checkpoint_predictions:
                for iter_num, preds in all_checkpoint_predictions[task_id].items():
                    submission_timeline.append({'task_idx': task_idx, 'task_id': task_id, 'iter': iter_num, 'predictions': preds})
        
        submission_timeline.sort(key=lambda x: (x['task_idx'], x['iter']))
        
        temp_submission = {}
        for event in submission_timeline:
            temp_submission[event['task_id']] = event['predictions']
            
            for tid in task_ids:
                if tid not in temp_submission:
                     if tid in challenges_dict and 'test' in challenges_dict[tid]:
                         num_test_cases = len(challenges_dict[tid]['test'])
                         default_preds = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(num_test_cases)]
                         temp_submission[tid] = default_preds

            filename = f"submission_task{event['task_idx']}_{event['task_id']}_iter{event['iter']:04d}.json"
            filepath = os.path.join(intermediate_submissions_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(temp_submission, f, indent=4)

    all_results = []
    
    submission_files = sorted(glob.glob(os.path.join(intermediate_submissions_dir, "*.json")))
    if os.path.exists(final_submission_path):
        submission_files.append(final_submission_path)

    if not submission_files:
        print("  No submission files found to visualize.")
        return

    pdf_path = os.path.join(output_dir_path, "visualization_all.pdf")
    combined_pdf = PdfPages(pdf_path)
    
    try:
        for sub_file_path in submission_files:
            submission_name = os.path.basename(sub_file_path)
            print(f"  Visualizing {submission_name}...")
            with open(sub_file_path, 'r') as f:
                submission_dict_ind = json.load(f)

            results_filename = f"results_{submission_name.replace('.json', '.md')}"
            pdf_title = f"Results for: {submission_name}"
            
            result = evaluate_submission(
                submission_dict=submission_dict_ind,
                solutions_dict=solutions_dict,
                challenges_dict=challenges_dict,
                output_dir_path=output_dir_path,
                visualize=True,
                pdf_object=combined_pdf,
                results_filename=results_filename,
                pdf_title=pdf_title
            )
            result['submission_file'] = submission_name
            all_results.append(result)

        combined_pdf.close()
        print(f"\nCombined visualization saved to {pdf_path}")

        summary_path = os.path.join(output_dir_path, "results_all_summary.md")
        with open(summary_path, 'w') as f:
            f.write("# Overall Training Progression\n\n")
            f.write("| Submission File | Overall Union Pixel Accuracy | Fully Correct Tasks |\n")
            f.write("|:---|:---:|:---:|\n")
            for res in all_results:
                f.write(f"| {res['submission_file']} | {res['overall_pixel_accuracy']:.4f} | {res['num_fully_correct_tasks']} |\n")
        print(f"Overall summary saved to {summary_path}")

    except Exception as e:
        print(f"  An error occurred during training progression visualization: {e}")

def evaluate_submission(
    submission_dict: dict,
    solutions_dict: dict,
    challenges_dict: dict,
    output_dir_path: str,
    visualize: bool = False,
    pdf_object: PdfPages = None,
    results_filename: str = "results.md",
    pdf_title: str = None
) -> dict:
    """
    Compare an ARC-AGI submission against the ground truth solutions, compute per-task correctness and pixel accuracy,
    and optionally generate a PDF visualizing inputs, predictions, and ground truths.

    Parameters
    ----------
    submission_dict : dict
        Mapping from task_id to a list of dicts, each with keys "attempt_1" and "attempt_2" (predicted output grids).
    solutions_dict : dict
        Mapping from task_id to a list of ground-truth output grids or list of dicts with key "output".
    challenges_dict : dict
        Mapping from task_id to challenge data; used to extract test input grids (each challenge entry has a "test" field,
        which is a list of dicts with key "input").
    output_dir_path : str
        Path to the directory where results.csv and visualization.pdf will be saved.
    visualize : bool, optional (default=False)
        If True, generate a "visualization.pdf" in output_dir_path showing, for each test input:
        - the input grid
        - both predicted attempts
        - the ground truth grid
        Padded regions will be shown in grey for shape mismatches.
    pdf_object : PdfPages, optional
        An existing PdfPages object to write visualization to. If None, a new PDF is created.
    results_filename : str, optional
        Filename for the output results markdown file.
    pdf_title : str, optional
        If provided, a title page with this string is added to the PDF.

    Returns
    -------
    results : dict
        A dictionary containing:
        - mismatched_submission_ids: list of task IDs in submission but not in solutions
        - mismatched_solution_ids: list of task IDs in solutions but not in submission
        - num_total_tasks_in_both: count of tasks present in both
        - num_fully_correct_tasks: number of tasks where fraction_correct == 1.0
        - fully_correct_ids: list of task IDs that are fully correct
        - overall_pixel_accuracy: total matched pixels / total pixels (float)
        - per_task: dict mapping task_id to a dict with:
            - fraction_correct: fraction of test outputs exactly correct (0.0 to 1.0)
            - per_test_pixel_acc: list of per-test-input pixel accuracies (floats)
            - task_avg_pixel_acc: average pixel accuracy across all test inputs
    """

    # 1) Identify mismatched task IDs
    submission_ids = set(submission_dict.keys())
    solution_ids = set(solutions_dict.keys())

    mismatched_submission_ids = sorted(list(submission_ids - solution_ids))
    mismatched_solution_ids = sorted(list(solution_ids - submission_ids))
    common_task_ids = sorted(list(submission_ids & solution_ids))

    # 2) Prepare colormap for visualization (-1 => grey, 0-9 => categorical)
    # Using colors from the provided style
    ARC_COLORS = [
        (0, 0, 0),                    # 0: black
        (0, 116/255, 217/255),      # 1: blue
        (255/255, 65/255, 54/255),  # 2: red
        (46/255, 204/255, 64/255),  # 3: green
        (255/255, 220/255, 0/255),  # 4: yellow
        (170/255, 170/255, 170/255),# 5: grey
        (240/255, 22/255, 230/255), # 6: magenta/pink
        (255/255, 133/255, 27/255), # 7: orange
        (127/255, 219/255, 255/255),# 8: cyan
        (135/255, 15/255, 35/255),  # 9: dark red
    ]
    # Padding color (for -1 grid value). The docstring mentions grey for padding.
    PADDING_COLOR = (255/255, 255/255, 255/255) # white

    # Colormap for values -1 (padding) through 9.
    # We shift values by +1 for plotting, so -1 -> 0, 0 -> 1, ..., 9 -> 10.
    cmap_colors = [PADDING_COLOR] + ARC_COLORS
    cmap = ListedColormap(cmap_colors)
    # Bounds for values 0..11 (corresponds to original values -1..9)
    bounds = list(range(len(cmap_colors) + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    # Prepare PDF if visualization requested
    pdf = None
    own_pdf_management = False
    if visualize:
        if pdf_object is None:
            pdf_path = os.path.join(output_dir_path, "visualization.pdf")
            pdf = PdfPages(pdf_path)
            own_pdf_management = True
        else:
            pdf = pdf_object

        if pdf_title and pdf:
            # Create a title page for this section
            fig = plt.figure(figsize=(8, 11))
            fig.text(0.5, 0.5, pdf_title, ha='center', va='center', fontsize=20, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

        print("Please wait visualising...")
        print("This might take upto 2 minutes if there are a large number of tasks.")

    # 3) Aggregators for overall pixel counts
    total_matched_pixels = 0
    total_pixels_overall = 0

    # 4) Per-task results storage
    per_task = {}

    for task_id in common_task_ids:
        # Extract ground-truth output grids
        sol_list = solutions_dict[task_id]
        ground_truths = []
        for elem in sol_list:
            if isinstance(elem, dict) and "output" in elem:
                ground_truths.append(elem["output"])
            else:
                # Assume elem is directly a grid (list of lists)
                ground_truths.append(elem)

        # Extract submission attempts
        pred_list = submission_dict[task_id]
        # Each pred_list element is a dict with 'attempt_1' and 'attempt_2'
        # Ensure lengths match
        num_tests_truth = len(ground_truths)
        num_tests_pred = len(pred_list)
        num_tests = min(num_tests_truth, num_tests_pred)

        # Extract input grids from challenges for visualization
        challenge_entry = challenges_dict.get(task_id, {})
        test_items = challenge_entry.get("test", [])
        test_inputs = []
        for test_item in test_items:
            if isinstance(test_item, dict) and "input" in test_item:
                test_inputs.append(test_item["input"])
            else:
                # Unexpected format
                test_inputs.append(None)

        # 5) Per-test statistics
        per_test_pixel_acc = []
        num_exact_correct = 0  # count of test outputs fully matched

        for idx in range(num_tests):
            true = ground_truths[idx]
            pred_entry = pred_list[idx]
            pred1 = pred_entry.get("attempt_1", [])
            pred2 = pred_entry.get("attempt_2", [])

            # Check for exact full match
            exact1 = (pred1 == true)
            exact2 = (pred2 == true)
            if exact1 or exact2:
                num_exact_correct += 1

            # NEW: Compute pixel accuracy using the union logic
            matches1, total_cells1 = calculate_union_accuracy(pred1, true)
            matches2, total_cells2 = calculate_union_accuracy(pred2, true)

            pixel_acc1 = matches1 / total_cells1 if total_cells1 > 0 else 0.0
            pixel_acc2 = matches2 / total_cells2 if total_cells2 > 0 else 0.0

            # Take the higher of the two accuracies for this test case
            best_pixel_acc_for_test = max(pixel_acc1, pixel_acc2)
            per_test_pixel_acc.append(best_pixel_acc_for_test)

            # Accumulate best matches and their corresponding total cells for overall accuracy
            if pixel_acc1 >= pixel_acc2:
                total_matched_pixels += matches1
                total_pixels_overall += total_cells1
            else:
                total_matched_pixels += matches2
                total_pixels_overall += total_cells2

        # 9) Summarize per-task
        fraction_correct = num_exact_correct / num_tests if num_tests > 0 else 0.0
        # per_test_pixel_acc already contains the best union accuracy for each test
        task_avg_pix = sum(per_test_pixel_acc) / len(per_test_pixel_acc) if per_test_pixel_acc else 0.0

        per_task[task_id] = {
            "fraction_correct": fraction_correct,
            "per_test_pixel_acc": per_test_pixel_acc,
            "task_avg_pixel_acc": task_avg_pix
        }

    # Close PDF if opened
    if visualize and pdf is not None: # Ensure pdf was actually created and is not None
        # Sort task IDs by descending task_avg_pixel_acc for visualization
        sorted_task_ids = sorted(
            per_task.keys(),
            key=lambda tid: per_task[tid]["task_avg_pixel_acc"],
            reverse=True
        )

        # Loop through sorted tasks to generate visualization pages
        for task_id in sorted_task_ids:
            # (a) Extract ground‐truths & preds
            sol_list = solutions_dict[task_id]
            ground_truths = []
            for elem in sol_list:
                if isinstance(elem, dict) and "output" in elem:
                    ground_truths.append(elem["output"])
                else:
                    ground_truths.append(elem)

            pred_list = submission_dict[task_id]

            # (b) Extract test inputs
            challenge_entry = challenges_dict.get(task_id, {})
            test_items      = challenge_entry.get("test", [])
            test_inputs     = []
            for test_item in test_items:
                if isinstance(test_item, dict) and "input" in test_item:
                    test_inputs.append(test_item["input"])
                else:
                    test_inputs.append(None)

            num_tests = min(len(ground_truths), len(pred_list))

            # (c) For each test, re‐pad & plot
            for idx in range(num_tests):
                true       = ground_truths[idx]
                pred_entry = pred_list[idx]
                pred1      = pred_entry.get("attempt_1", [])
                pred2      = pred_entry.get("attempt_2", [])

                # Re-calculate pixel accuracies for display in titles
                # This is a slight redundancy, but keeps the plotting loop self-contained
                matches1, total_cells1 = calculate_union_accuracy(pred1, true)
                matches2, total_cells2 = calculate_union_accuracy(pred2, true)
                pixel_acc1 = matches1 / total_cells1 if total_cells1 > 0 else 0.0
                pixel_acc2 = matches2 / total_cells2 if total_cells2 > 0 else 0.0

                true_h = len(true)
                true_w = len(true[0]) if true_h > 0 else 0
                p1_h = len(pred1)
                p1_w = len(pred1[0]) if p1_h > 0 else 0
                p2_h = len(pred2)
                p2_w = len(pred2[0]) if p2_h > 0 else 0

                # Determine common bounding box for plotting all 4 grids (input, pred1, pred2, true)
                inp_h, inp_w = (0,0)
                if idx < len(test_inputs) and test_inputs[idx] is not None:
                    inp_h = len(test_inputs[idx])
                    inp_w = len(test_inputs[idx][0]) if inp_h > 0 else 0

                common_h_plot = max(true_h, p1_h, p2_h, inp_h)
                common_w_plot = max(true_w, p1_w, p2_w, inp_w)

                true_p = _pad_grid(true, common_h_plot, common_w_plot)
                p1_p = _pad_grid(pred1, common_h_plot, common_w_plot)
                p2_p = _pad_grid(pred2, common_h_plot, common_w_plot)

                input_grid_to_plot = [[-1] * common_w_plot for _ in range(common_h_plot)] # Default to empty padded
                if idx < len(test_inputs) and test_inputs[idx] is not None:
                    input_grid_to_plot = _pad_grid(test_inputs[idx], common_h_plot, common_w_plot)

                fig, axes = plt.subplots(2, 2, figsize=(6, 6))
                titles = [
                    "Input",
                    f"Attempt 1 (Acc: {pixel_acc1:.2%})",
                    f"Attempt 2 (Acc: {pixel_acc2:.2%})",
                    "Ground Truth"
                ]
                grids  = [input_grid_to_plot, p1_p, p2_p, true_p]
                for ax, title, grid_to_plot in zip(axes.flatten(), titles, grids):
                    if not grid_to_plot or not grid_to_plot[0]:
                        ax.set_title(title)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Shift values so padding (-1) becomes 0, and colors match cmap
                    shifted = [[cell + 1 for cell in row] for row in grid_to_plot]
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

                plt.suptitle(f"Task {task_id} | Test #{idx+1}", fontsize=10)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
        if own_pdf_management:
            pdf.close() # Close PDF after the entire visualization loop

    # 10) Compute overall stats
    num_total_tasks_in_both = len(common_task_ids)
    fully_correct_ids = [
        tid for tid, info in per_task.items() if info["fraction_correct"] == 1.0
    ]
    num_fully_correct_tasks = len(fully_correct_ids)
    overall_pixel_accuracy = (
        total_matched_pixels / total_pixels_overall if total_pixels_overall > 0 else 0.0
    )

    # 11) Write summary to results.md
    results_file_path = os.path.join(output_dir_path, results_filename)
    with open(results_file_path, "w") as f:
        f.write(f"# ARC-AGI Evaluation Results\n\n") # Enhanced header
        f.write(f"**Submission File:** `{os.path.basename(results_filename).replace('results_', '').replace('.md', '.json')}`\n\n")
        f.write(f"**Total tasks evaluated:** {num_total_tasks_in_both}\n")
        if mismatched_submission_ids:
            f.write(f"**Tasks in submission but not in solutions:** {', '.join(mismatched_submission_ids)}\n")
        if mismatched_solution_ids:
            f.write(f"**Tasks in solutions but not in submission:** {', '.join(mismatched_solution_ids)}\n")
        f.write(f"**Number of fully-correct tasks (100% exact match):** {num_fully_correct_tasks}\n")
        if fully_correct_ids:
            f.write(f"**List of fully-correct task IDs:** {', '.join(fully_correct_ids)}\n")
        f.write(f"**Overall Union Pixel Accuracy (across all test cases):** {overall_pixel_accuracy:.4f}\n\n")

        f.write("## Per-Task Details\n\n") # New section header
        # 12) Build and write markdown table
        header = ["Task ID", "Fraction Correct (Exact)", "Avg. Union Pixel Acc.", "Individual Test Accuracies"] # Improved headers
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|:---|:---------------------|:--------------------|:-------------------------|\n") # Markdown table header separator
        rows_to_write = []
        # Sort by fraction_correct first, then by task_avg_pixel_acc
        sorted_task_items = sorted(
            per_task.items(),
            key=lambda kv: (kv[1]["fraction_correct"], kv[1]["task_avg_pixel_acc"]),
            reverse=True
        )
        for tid, info in sorted_task_items:
            frac = f"{info['fraction_correct']:.4f}"
            avg_pix = f"{info['task_avg_pixel_acc']:.4f}"
            per_list = "; ".join(f"{acc:.4f}" for acc in info["per_test_pixel_acc"]) # Use "; " for readability
            rows_to_write.append([tid, frac, avg_pix, per_list])

        for row in rows_to_write:
            f.write("| " + " | ".join(row) + " |\n")

    print(f"\nEvaluation complete.")
    print(f"Evaluation results saved to: {os.path.join(output_dir_path, results_filename)}")

    # 14) Return dictionary of results
    return {
        "mismatched_submission_ids": mismatched_submission_ids,
        "mismatched_solution_ids": mismatched_solution_ids,
        "num_total_tasks_in_both": num_total_tasks_in_both,
        "num_fully_correct_tasks": num_fully_correct_tasks,
        "fully_correct_ids": fully_correct_ids,
        "overall_pixel_accuracy": overall_pixel_accuracy,
        "per_task": per_task
    }


# --- Example Usage ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI submission and optionally generate visualizations.")
    parser.add_argument("--submission_file", type=str, required=True, help="Path to the submission JSON file.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the directory containing challenges.json and solutions.json.")
    parser.add_argument("--visualize", "--visualise", action="store_true", help="Generate visualization.pdf.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    submission_dir = os.path.dirname(args.submission_file)
    output_dir_path = submission_dir if submission_dir else "." # Use current directory if submission_file is just a filename
    os.makedirs(output_dir_path, exist_ok=True)

    if args.visualize:
        print("Visualisation mode on")
    print(f"Loading submission from: {args.submission_file}")
    print(f"Loading dataset from: {args.dataset}")
    print(f"Output directory set to: {output_dir_path}")

    try:
        with open(args.submission_file) as f:
            submission_dict = json.load(f)
        with open(os.path.join(args.dataset, "solutions.json")) as f:
            solutions_dict = json.load(f)
        with open(os.path.join(args.dataset, "challenges.json")) as f:
            challenges_dict = json.load(f)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Could not find a required file.")
        print(f"Missing file: {e.filename}")
        print("Please ensure the provided file paths are correct.")
        exit()
    except json.JSONDecodeError as e:
        print(f"\nFATAL ERROR: Could not parse JSON in one of the files.")
        print(f"Error: {e}")
        exit()


    # Call the evaluator function
    results = evaluate_submission(
        submission_dict=submission_dict,
        solutions_dict=solutions_dict,
        challenges_dict=challenges_dict,
        output_dir_path=output_dir_path,
        visualize=args.visualize
    )

    

    if args.visualize:
        print(f"Visualisation complete.")
        print(f"Visualization PDF saved to: {os.path.join(output_dir_path, 'visualization.pdf')}")