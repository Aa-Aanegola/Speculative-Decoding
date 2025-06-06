# HPML Project: SpecAdapt - Adaptive Semantic Prompt Routing for Speculative Decoding 
## Team Information
- **Team Name**: Hype
- **Members**:
  - Aakash Aanegola (aa5506)
  - Nico Bykhovsky (nb3227)
  - Aryamaan Saha (as7482)

---
## 1. Problem Statement
Large Language Model (LLM) inference faces challenges in achieving both low latency and high throughput, especially for diverse incoming prompts. Speculative decoding techniques offer significant speedups, but their performance varies depending on the specific method and the characteristics of the input prompt. Choosing a single, static decoding strategy is sub-optimal across a heterogeneous stream of requests. Our project introduces SpecAdapt, a prompt-aware speculative decoding router. It embeds incoming prompts, clusters them using KMeans over MiniLM embeddings, and routes them to the best-performing decoding strategy for their semantic cluster. This strategy improves throughput by dynamically adapting to the nature of each prompt. We evaluate methods across different inference backends, including vLLM, to understand the interplay between speculative decoding and engine-level optimizations.

---
## 2. Model Description
The project primarily utilizes pre-trained transformer-based Large Language Models for the target and draft models in speculative decoding, as well as a Sentence Transformer model for prompt embedding.

-   **Target LLMs:**  Meta-Llama-3.1-8B-Instruct
-   **Draft LLMs:** Meta-Llama-3.2-1B-Instruct
-   **Embedding Model:** `all-MiniLM-L6-v2`, a lightweight Sentence Transformer model, is used to generate semantic embeddings of prompts.
-   **Clustering Model:** `scikit-learn`'s KMeans is used for clustering prompt embeddings.

**Frameworks & Libraries:** The project is implemented using Python, primarily leveraging:
-   PyTorch for model loading and inference.
-   Hugging Face Transformers for standard model architectures, tokenization, and generation utilities.
-   vLLM inference engine for benchmarking performance under different backend optimizations.
-   Specific libraries for advanced speculative decoding implementations (EAGLE, FSD, potential future integrations like Umbrella, vLLM).
-   `sentence-transformers` for the embedding model.
-   `scikit-learn` for clustering.
---
## 3. Final Results Summary

Our primary contribution is a prompt-aware routing system that selects the best speculative decoding method per semantic cluster, based on benchmarks run using the standard PyTorch/Hugging Face backend. We also conducted comparative benchmarks on vLLM to understand backend performance. For detailed performance comparisons across methods and backends, see the tables below (speedups are relative to naive decoding on standard PyTorch):

**Harmonic Mean Speedups (Overall Averages):**

| Method               | Harmonic Mean Speedup |
| :------------------- | :-------------------- |
| Spec-Dec (vLLM)      | **1.87**              |
| Eagle1 (vLLM)        | 1.81                  |
| Spec-Dec (Std)       | 1.52                  |
| FSD (Std)            | 1.34                  |
| Eagle2 (Std)         | 1.33                  |
| Naive (vLLM)         | 1.13                  |

**Cluster-wise Speedup Comparison:**

| Cluster | FSD (Std) | Spec-Dec (Std) | Eagle2 (Std) | Naive (vLLM) | Spec-Dec (vLLM) | Eagle1 (vLLM) |
| :------ | :-------- | :------------- | :----------- | :----------- | :-------------- | :------------ |
| 0       | 1.51      | 1.53           | **1.84**     | 1.13         | 1.73            | 1.74          |
| 1       | 1.60      | 1.83           | 1.58         | 1.13         | 2.13            | **2.20**      |
| 2       | 1.66      | 1.75           | 1.71         | 1.14         | **1.86**        | 1.77          |
| 3       | 1.60      | 1.37           | 1.36         | 1.13         | 1.74            | **1.76**      |
| 4       | 0.52      | 1.24           | 0.52         | 1.13         | **1.83**        | 1.80          |
| 5       | 1.73      | 1.52           | 1.78         | 1.13         | **1.91**        | 1.83          |
| 6       | 1.58      | 1.51           | 1.27         | 1.13         | **1.88**        | 1.74          |
| 7       | 1.80      | 1.51           | 1.73         | 1.13         | **1.93**        | 1.86          |
| 8       | 1.59      | 1.58           | 1.71         | 1.13         | **1.92**        | 1.78          |
| 9       | 1.63      | 1.54           | 1.66         | 1.13         | **1.82**        | 1.75          |

*(Note:  Bold indicates the highest speedup observed within that cluster across all tested methods.)*

---
## 4. Reproducibility Instructions

### A. Requirements
Install core dependencies:
```bash
pip install -r requirements.txt
```

Note: Some speculative decoding methods (EAGLE, FSD) require special installations outside of `requirements.txt`. Refer to their respective GitHub repositories for instructions if you plan to use these methods.
- EAGLE: [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
- FSD: [https://github.com/maxholsman/fsd](https://github.com/maxholsman/fsd)
- FastChat (needed for EAGLE templating): `pip install fschat`

### B. Wandb Dashboard

Benchmark results and routing performance metrics were logged to Wandb.
View the logs for our experiments here: [https://wandb.ai/speculative-decoding](https://wandb.ai/speculative-decoding). The projects used are `full-profiles` and `vllm-routing`.

<!-- # SpecAdapt
SpecAdapt is a framework designed to optimize speculative decoding models by providing tools for benchmarking, clustering, and routing. It includes scripts and utilities to evaluate model performance, group input prompts into clusters, and route prompts to the most suitable decoding model based on scoring mechanisms. The framework is organized into three main subdirectories: benchmarks/ for performance evaluation, clustering/ for analyzing and grouping input data, and routing/ for assigning prompts to models. By leveraging configurable scripts and profiling tools, SpecRouter enables efficient testing, analysis, and deployment of speculative decoding techniques. -->

<!-- ### Setup 
run `pip install -r requirements.txt` to install necessary packages. Note that EAGLE and FSD require special installations from [EAGLE](https://github.com/SafeAILab/EAGLE) and [FSD](https://github.com/maxholsman/fsd) -->

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

---

## Running the Web Application

The framework includes a simple Flask web application (routing/app.py) that provides an interactive chat interface leveraging the routing logic.

### To Launch the UI:

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   (Also install any model-specific ones like EAGLE/FSD.)

2. Navigate to the routing/ directory and run:

   ```bash
   python app.py
   ```

   The server runs on `http://0.0.0.0:5000`.

### Accessing the UI:

* **If running on your local machine::**
  Open `http://127.0.0.1:5000` or `http://localhost:5000` in your browser.

* **On a Remote Cluster (via SSH Port Forwarding):**
  Direct access to port 5000 on the remote server might be blocked by network configurations. If you're running `app.py` inside an interactive job on a compute node (e.g., after using `srun --pty`), you need to forward traffic from your local machine to that compute node. Open a terminal on your **local machine** and use the ssh -L command to create a tunnel:

  ```bash
  ssh -N -L 5000:<compute_node_hostname>:5000 <username>@<cluster_login_node>
  ```

  * Replace `<compute_node_hostname>` with the name of the interactive node (e.g., the one allocated via `srun`).
  * Replace `<username>` and `<cluster_login_node>` with your login credentials.
  * Once connected, open `http://localhost:5000` in your local browser.

After launching the app and forwarding the port, you should see a UI like this:
<!-- ![Chat UI Preview](routing/static/ui-preview.png) -->
<img src="routing/static/ui-preview.png" alt="Chat UI Preview" width="600"/>

