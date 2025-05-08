# SpecRouter
SpecRouter is a framework designed to optimize speculative decoding models by providing tools for benchmarking, clustering, and routing. It includes scripts and utilities to evaluate model performance, group input prompts into clusters, and route prompts to the most suitable decoding model based on scoring mechanisms. The framework is organized into three main subdirectories: benchmarks/ for performance evaluation, clustering/ for analyzing and grouping input data, and routing/ for assigning prompts to models. By leveraging configurable scripts and profiling tools, SpecRouter enables efficient testing, analysis, and deployment of speculative decoding techniques.

### Setup 
run `pip install -r requirements.txt` to install necessary packages. Note that EAGLE and FSD require special installations from [EAGLE](https://github.com/SafeAILab/EAGLE) and [FSD](https://github.com/maxholsman/fsd)

## Benchmarks Subdirectory

The `benchmarks/` subdirectory contains resources and scripts for evaluating the performance of speculative decoding models. Below is an overview of the key files:

### `uc_profile`
This file provides profiling utilities for speculative decoding. It includes tools to measure performance metrics such as latency, throughput, and memory usage during model execution.

### `specbench.py`
This script defines and runs benchmark tests for speculative decoding models. It includes:
- Predefined test cases for various decoding scenarios.
- Configurable parameters to customize benchmarks.
- Output of detailed performance statistics for analysis.

### Usage
1. Ensure all dependencies are installed.
2. The important configs are of the type `model-name.yml` (for example `fsd.yml`, `naive.yml`, `eagle2.yml` etc.). 
2. Run `specbench.py` to execute benchmarks:
    ```bash
    python specbench.py --yaml <config-path> --dataset <dataset-path> --output_dir <out_path>
    python uc_profile.py --yaml <config-path> --dataset <dataset-path> --output_dir <out_path>
    ```
3. Review the output for performance insights.

### Notes
- Modify `uc_profile` to customize profiling behavior.
- Use this directory to profile and optimize speculative decoding implementations.



## Clustering Subdirectory

The `clustering/` subdirectory contains tools and scripts for analyzing and grouping input prompts. We use this to determine overall clusters to evaluate different speculative decoding techniques that are then used by the router. 

### `process_ultrachat`
This script processes and clusters conversational data from the UltraChat dataset. It includes:
- Preprocessing steps to clean and normalize the data.
- Clustering algorithms to group similar conversations.
- Visualization tools to analyze clustering results.

### Usage 
2. Run `process_ultrachat` to perform clustering:
    ```bash
    python process_ultrachat.py --yaml ../configs/clustering.yml
    ```
3. Review the output for clustering insights and visualizations.

### Notes
- Adjust the number of clusters using the `clusters` parameter in the config to refine results.


## Routing Subdirectory

The `routing/` subdirectory contains the core logic for routing input prompts to the most suitable speculative decoding model. This is achieved by leveraging scoring mechanisms and routing algorithms.

### `main.py`
This script serves as the entry point for the routing process. It includes:
- Initialization of routing configurations and models.
- Integration with scoring mechanisms to evaluate input prompts.
- Execution of the routing logic to assign prompts to appropriate models.

### `compute_scores.py`
This script computes scores for input prompts based on predefined metrics. It includes:
- Feature extraction from input prompts.
- Scoring algorithms to evaluate prompt characteristics.
- Output of scores used by the router for decision-making.

### Usage
1. Run `main.py` to perform routing:
    ```bash
    python main.py --yaml ../configs/routing.yml --mode `interactive | benchmark`
    ```
2. Optionally, run `compute_scores.py` to generate scores independently:
    ```bash
    python compute_scores.py --yaml ../configs/scoring.yml
    ```