import json
import yaml
import argparse
import pandas as pd
from scipy.stats import hmean
import sys
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate harmonic mean speedups based on TPS from JSON results using a YAML config.')
    parser.add_argument('--yaml', type=str, required=True, help='YAML configuration file path')
    args = parser.parse_args()

    try:
        with open(args.yaml, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.yaml}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}")
        sys.exit(1)

    methods_config = config.get('methods')
    output_path = config.get('output_path')

    if not methods_config:
        print("Error: Config file must contain a 'methods' list.")
        sys.exit(1)
    if not output_path:
        print("Error: Config file must contain an 'output_path'.")
        sys.exit(1)
    if not isinstance(methods_config, list) or not methods_config:
         print("Error: 'methods' in config must be a non-empty list.")
         sys.exit(1)

    baseline_item = methods_config[0]
    baseline_name = baseline_item.get('name')
    baseline_file = baseline_item.get('file')

    if not baseline_name or not baseline_file:
         print(f"Error: The first method in the list (baseline) must have both 'name' and 'file': {baseline_item}")
         sys.exit(1)

    print(f"Identified baseline method: '{baseline_name}' from '{baseline_file}'")

    all_data = []
    successfully_loaded_methods = []

    print("Loading data from JSON files...")
    for item in methods_config:
        method_name = item.get('name')
        file_path = item.get('file')
        if not method_name or not file_path:
            print(f"Warning: Skipping incomplete method entry in config: {item}")
            continue

        try:
            with open(file_path, 'r') as f:
                method_data = json.load(f)
                for entry in method_data:
                    entry['method'] = method_name
                all_data.extend(method_data)
                successfully_loaded_methods.append(method_name)
            print(f"Successfully loaded data for method '{method_name}' from '{file_path}'.")

        except FileNotFoundError:
            print(f"Error: File not found for method '{method_name}' at '{file_path}'. Skipping this method.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_path}' for method '{method_name}'. Skipping this method.")
        except Exception as e:
            print(f"An unexpected error occurred loading data for method '{method_name}': {e}")

    if not successfully_loaded_methods:
        print("Error: No data was successfully loaded from any method files.")
        sys.exit(1)
    if baseline_name not in successfully_loaded_methods:
         print(f"Error: Baseline method '{baseline_name}' file was not loaded successfully. Cannot proceed with calculations.")
         sys.exit(1)

    print("Preparing DataFrame...")
    df = pd.DataFrame(all_data)

    required_cols = ['id', 'cluster', 'tokens_per_second', 'method']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: DataFrame is missing required columns: {missing}. Please check input JSON files.")
        sys.exit(1)

    df_filtered = df[df['tokens_per_second'] > 0].copy()

    if df_filtered.empty:
        print("Error: No data points with positive tokens_per_second found across all files after filtering.")
        sys.exit(1)

    df_baseline = df_filtered[df_filtered['method'] == baseline_name].rename(columns={'tokens_per_second': 'tps_baseline', 'cluster': 'cluster_base'})
    df_methods = df_filtered[df_filtered['method'] != baseline_name].rename(columns={'tokens_per_second': 'tps_method', 'cluster': 'cluster_method'})

    if df_baseline.empty:
         print(f"Error: No baseline data with positive TPS found for method '{baseline_name}'. Cannot proceed.")
         sys.exit(1)

    print("Calculating speedups and grouping by cluster...")

    cluster_raw_results = defaultdict(lambda: defaultdict(list))

    comparison_methods = [m for m in successfully_loaded_methods if m != baseline_name]

    baseline_clusters = sorted(df_baseline['cluster_base'].unique(), key=int)

    for cluster in baseline_clusters:
        baseline_cluster_df = df_baseline[df_baseline['cluster_base'] == cluster].copy()

        for method in comparison_methods:
            method_cluster_df = df_methods[
                 (df_methods['method'] == method) &
                 (df_methods['cluster_method'] == cluster)
             ].copy()

            merged_df = pd.merge(
                baseline_cluster_df[['id', 'tps_baseline']],
                method_cluster_df[['id', 'tps_method']],
                on='id',
                how='inner'
            )

            if not merged_df.empty:
                speedups = merged_df['tps_method'] / merged_df['tps_baseline']
                cluster_raw_results[cluster][method].extend(speedups.tolist())

    print("Calculating harmonic means and formatting output...")
    final_results = {}

    sorted_clusters = sorted(cluster_raw_results.keys(), key=int)

    all_methods_in_results = set()
    for method_data in cluster_raw_results.values():
        all_methods_in_results.update(method_data.keys())
    sorted_methods_in_output = sorted(list(all_methods_in_results))

    for cluster in sorted_clusters:
        final_results[str(cluster)] = {}

        for method in sorted_methods_in_output:
            speedup_list = cluster_raw_results.get(cluster, {}).get(method, [])

            harmonic_mean_speedup = 0.0
            if speedup_list:
                try:
                    harmonic_mean_speedup = hmean(speedup_list)
                except ValueError as e:
                    print(f"Warning: Could not calculate harmonic mean for Cluster {cluster}, Method '{method}' due to ValueError: {e}. Speedups list count: {len(speedup_list)}")
                    harmonic_mean_speedup = 0.0

            final_results[str(cluster)][method] = float(harmonic_mean_speedup)

    print(f"Saving results to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print("Calculation complete. Results saved.")
    except IOError as e:
        print(f"Error: Could not write output file {output_path}: {e}")
        sys.exit(1)