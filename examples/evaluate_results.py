import os
import json
import numpy as np


def load_results(data_dir):
    methods = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    all_data = {}

    for method in methods:
        method_dir = os.path.join(data_dir, method)
        method_results = {}
        for filename in os.listdir(method_dir):
            if filename.endswith('.json'):
                map_idx = filename.split('.')[0]
                with open(os.path.join(method_dir, filename), 'r') as f:
                    method_results[map_idx] = json.load(f)
        all_data[method] = method_results
    return all_data


def evaluate(data_dir, baseline_method):
    all_data = load_results(data_dir)

    if baseline_method not in all_data:
        print(f"Error: Baseline method '{baseline_method}' not found in data.")
        return

    baseline_results = all_data[baseline_method]
    map_indices = sorted(baseline_results.keys(), key=int)

    # Calculate baseline averages first
    baseline_metrics = {m: [] for m in ['path_length', 'total_nodes_expanded', 'cumulative_risk', 'hard_collisions']}
    for idx in map_indices:
        for m in baseline_metrics.keys():
            baseline_metrics[m].append(baseline_results[idx][m])

    avg_baseline = {m: np.mean(baseline_metrics[m]) for m in baseline_metrics.keys()}

    metrics = ['path_length', 'total_nodes_expanded', 'cumulative_risk', 'hard_collisions']
    summary = []

    for method, results in all_data.items():
        method_metrics = {m: [] for m in metrics}
        successes = 0
        total_maps = 0

        for idx in map_indices:
            if idx in results:
                total_maps += 1
                res = results[idx]
                if res['success']:
                    successes += 1

                for m in metrics:
                    method_metrics[m].append(res[m])

        # Calculate averages for this method
        avg_method = {m: np.mean(method_metrics[m]) if method_metrics[m] else 0 for m in metrics}

        method_summary = {
            'Method': method,
            'SuccessRate': (successes / total_maps * 100) if total_maps > 0 else 0,
            'Avg_PathLen': avg_method['path_length'],
            'Avg_Nodes': avg_method['total_nodes_expanded'],
            'Avg_Risk': avg_method['cumulative_risk'],
            'Avg_Collisions': avg_method['hard_collisions'],
        }

        # Add improvement metrics based on the RATIO OF AVERAGES
        for m in metrics:
            base_val = avg_baseline[m]
            method_val = avg_method[m]
            if base_val != 0:
                imp = (base_val - method_val) / base_val * 100
                method_summary[f'Imp_{m}_%'] = imp
            else:
                method_summary[f'Imp_{m}_%'] = 0.0

        summary.append(method_summary)


    # Only include and sort these specific methods
    method_order = [
        "whole_map_original_binary_hard",
        "whole_map_new_binary",
        "sliding_window_new_binary"
    ]

    sorted_summary = []
    # Add ONLY these methods in order
    for m_name in method_order:
        for row in summary:
            if row['Method'] == m_name:
                sorted_summary.append(row)
                break

    return sorted_summary

def print_table(summary):
    if not summary:
        return

    headers = summary[0].keys()
    # Find max width for each column
    widths = {h: len(h) for h in headers}
    for row in summary:
        for h in headers:
            val = f"{row[h]:.2f}" if isinstance(row[h], (float, np.float64, np.float32)) else str(row[h])
            widths[h] = max(widths[h], len(val))

    # Print header
    header_str = " | ".join(h.ljust(widths[h]) for h in headers)
    print(header_str)
    print("-" * len(header_str))

    # Print rows
    for row in summary:
        row_str = " | ".join(
            (f"{row[h]:.2f}" if isinstance(row[h], (float, np.float64, np.float32)) else str(row[h])).ljust(widths[h])
            for h in headers)
        print(row_str)


if __name__ == "__main__":
    DATA_256 = "examples/data/256"
    BASELINE = "whole_map_original_binary_hard"

    print(f"Evaluating results in {DATA_256} against baseline: {BASELINE}")
    summary = evaluate(DATA_256, BASELINE)

    if summary:
        print("\n--- Summary Table ---")
        print_table(summary)

        # Save to CSV manually
        output_csv = "evaluation_summary_256.csv"
        headers = summary[0].keys()
        with open(output_csv, 'w') as f:
            f.write(",".join(headers) + "\n")
            for row in summary:
                line = ",".join(str(row[h]) for h in headers)
                f.write(line + "\n")
        print(f"\nSummary saved to {output_csv}")
