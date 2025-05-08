import json
import yaml
import argparse
from collections import defaultdict
import pandas as pd
from scipy.stats import hmean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Profiler')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    args = parser.parse_args()

    with open(args.yaml, 'r') as f:
        config = yaml.safe_load(f)

    if 'baseline' not in [item['name'] for item in config['methods']]:
        raise ValueError("Baseline method not found in config methods.")
    
    df = []
    
    for item in config['methods']:
        with open(item['file'], 'r') as f:
            data = [{**it, 'method': item['name']} for it in json.load(f)]
            df.extend(data)
    
    df = pd.DataFrame(df)
    

    baseline_name = 'baseline'
    clusters = df['cluster'].unique()
    methods = [m for m in df['method'].unique() if m != baseline_name]

    cluster_speedup_dict = defaultdict(dict)

    for cluster in clusters:
        baseline_times = df[(df['method'] == baseline_name) & (df['cluster'] == cluster)].sort_values('id')
        
        for method in methods:
            method_times = df[(df['method'] == method) & (df['cluster'] == cluster)].sort_values('id')
            
            merged = pd.merge(baseline_times, method_times, on='id', suffixes=('_base', '_method'))
            
            if not merged.empty:
                speedups = (merged['total_time_base'] / merged['total_time_method']).tolist()
                harmonic = hmean(speedups)
                cluster_speedup_dict[cluster][method] = harmonic

    cluster_speedup_dict = {
        int(cluster): {str(method): float(score) for method, score in methods.items()}
        for cluster, methods in cluster_speedup_dict.items()
    }
    max_per_cluster = {
        cluster: max(methods.items(), key=lambda x: x[1])
        for cluster, methods in cluster_speedup_dict.items()
    }
 
    
    print(cluster_speedup_dict)
    with open(config['output_path'], 'w') as f:
        json.dump(cluster_speedup_dict, f, indent=4)