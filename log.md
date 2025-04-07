# Experimentation Log

This file is meant to log the development progress of the shared task. Please log relevant experiments including a short motivation, the commands, and a summary of the obtained results or follow-up ideas.

**Usage:** If you want to add content, please create a new section with the current date as heading (if not yet available), and structure your content e.g. as follows:

```markdown
## 2025-03-27

### CLEF Baseline: Learning Rate Optimization

- training the models on the English data and testing learning rates in the following range: 1e-3, 5e-4, 3e-4, 1e-4, for each learning rate we train 3 models with different seeds.
  - command: see `scripts/baselines_cluster/clef-baseline-lr-search.sh`
  - wandb (weights & biases) project: https://wandb.ai/tanikina/clef2-normalization-src_baselines (this is just an example!)
  - metric values:
    |         | val-official-METEOR | val-sampled-METEOR | combined-val-METEOR |
    |---------|---------------------|--------------------|---------------------|
    | lr-1e-3 | 0.555               | 0.600              | 0.578               |
    | lr-5e-4 | .....               | .....              | .....               |
    | lr-3e-4 | .....               | .....              | .....               |
    | lr-1e-4 | .....               | .....              | .....               |
  - outcome: the best learning rate is ...
```

## 2025-04-04

### CLEF Zero-Shot Baseline (LLM inference)

Using the following prompt template (from `baseline.yaml`):

```
You are an expert in misinformation detection and fact-checking. Your task is to identify the central claim in the given post while preserving its original language.

The central claim should meet the following criteria:
- **Verifiable**: It must be a factual assertion that can be checked against evidence.
- **Concise**: It should be a single, clear sentence that captures the main claim of the post.
- **Socially impactful**: It should be a statement that could influence public opinion, health, or policy.
- **Free from rhetorical elements**: Do not include opinions, rhetorical questions, or unnecessary context.
- **Preserve Original Language**: The output should be in the same language as the input post.

Output only the central claim without additional explanation or formatting.

Post: {post}

Central claim:
```

**Motivation:** To find the results using the LLMs out-of-the-box without any detailed prompt engineering (so far, will focus on prompt enginneering later). Only focused on smaller models, will run bigger LLMs, once I have access to stronger GPUs. The purpose is to have the performance of LLMs for further comparison and to see what are the issues with the generated responses.

Identified issues so far:
- Output language - in same cases, the output is not in the **original** language (Need to include the specific langauge in the prompt)
- Generated claims are longer
- Generated claims are not contextualized - sometimes the LLM do not include all the important information

#### Evaluation on the training set

The results on the train set across 13 languages:

| model                 |   ara |   deu |   eng |   fra |    hi |    mr |   msa |    pa |   pol |   por |   spa |    ta |   tha |
|-----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| gemma-3-4b-it         | 0.319 | 0.122 | 0.230 | 0.261 | 0.258 | 0.221 | 0.207 | 0.282 | 0.171 | 0.272 | 0.266 | 0.366 | 0.039 |
| Llama-3.1-8B-Instruct | 0.370 | 0.118 | 0.237 | 0.257 | 0.256 | 0.284 | 0.190 | 0.304 | 0.161 | 0.273 | 0.266 | 0.348 | 0.049 |
| Qwen2.5-7B-Instruct   | 0.377 | 0.110 | 0.229 | 0.258 | 0.223 | 0.234 | 0.194 | 0.269 | 0.163 | 0.278 | 0.270 | 0.342 | 0.063 |

Command: `python -m scripts.run_experiments`

#### Evaluation on the validation set

| model                       |   ara |   deu |   eng |   fra |    hi |    mr |   msa |    pa |   pol |   por |   spa |    ta |   tha |
|-----------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| gemma-3-4b-it               | 0.289 | 0.123 | 0.237 | 0.239 | 0.242 | 0.302 | 0.207 | 0.249 | 0.165 | 0.274 | 0.255 | 0.282 | 0.042 |
| gemma-3-12b-it              | 0.326 | 0.146 | 0.197 | 0.256 | 0.236 | 0.263 | 0.194 | 0.255 | 0.161 | 0.291 | 0.261 | 0.314 | 0.053 |
| gemma-3-27b-it              | 0.313 | **0.163** | 0.249 | **0.276** | 0.232 | 0.286 | **0.218** | 0.304 | 0.196 | **0.305** | **0.272** | **0.351** | 0.073 |
| Llama-3.1-8B-Instruct       | 0.338 | 0.135 | 0.242 | 0.247 | **0.248** | 0.288 | 0.190 | **0.335** | 0.140 | 0.277 | 0.255 | 0.302 | 0.048 |
| Llama-3.1-70B-Instruct      | 0.284 | 0.155 | **0.256** | 0.264 | 0.213 | 0.296 | 0.209 | 0.301 | **0.206** | 0.290 | 0.262 | 0.288 | 0.077 |
| Llama-3.3-70B-Instruct      | 0.323 | 0.139 | 0.248 | 0.247 | 0.236 | 0.272 | 0.196 | 0.306 | 0.204 | 0.293 | 0.268 | 0.307 | 0.072 |
| Qwen2.5-7B-Instruct         | **0.352** | 0.138 | 0.231 | 0.246 | 0.205 | 0.264 | 0.197 | 0.247 | 0.148 | 0.288 | 0.270 | 0.329 | **0.079** |
| Qwen2.5-72B-Instruct        | 0.343 | 0.134 | 0.244 | 0.254 | 0.228 | **0.312** | 0.193 | 0.324 | 0.152 | 0.260 | 0.246 | 0.310 | 0.072 |
| Mistral-Large-Instruct-2411 | 0.280 | 0.137 | 0.227 | 0.253 | 0.197 | 0.270 | 0.183 | 0.256 | 0.177 | 0.266 | 0.246 | 0.310 | 0.064 |