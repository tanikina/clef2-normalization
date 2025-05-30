## dfkinit2b at CheckThat! 2025:<br>Leveraging LLMs and Ensemble of Methods for Multilingual Claim Normalization

The task description is based on the official shared task repository: [https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task2](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task2)

Given a noisy, unstructured social media post, the task is to simplify it into a concise form.
This is a text generation task in which systems have to generate the normalized claims for the given social media posts.

The task comprises two settings:

- **Monolingual**: In this setup, the training, development, and test datasets are all provided for the specific language. The model is trained, validated, and tested using data exclusively from this single language, meaning that all the stages (training, validation, and testing) are confined to the same language. This setup ensures that the model learns language-specific patterns and structures. Languages: English, German, French, Spanish, Portuguese, Hindi, Marathi, Punjabi, Tamil, Arabic, Thai, Indonesian, and Polish.
- **Zero-shot**: In this case, only the test data is available for the specific language, and you are not provided with any training or development data for that language. You are free to use data from other languages for training your models, or you can choose to conduct a zero-shot experiment using LLMs, where the model is tested on the target language without being exposed to any training data. This approach evaluates how well the model can generalize to unseen languages. Languages: Dutch, Romanian, Bengali, Telugu, Korean, Greek, and Czech.

## Data

The data for the shared task can be found in the official GitLab repository: [https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task2](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task2). There should be three folders with train, dev and test data, and each split has subfolders for each language, in the csv format. In order to pre-process the data to remove duplicates and discard claim-post pairs with very low similarity you can run `python src/utils/prepare_data.py --target_lang=all`.

## Input Data Format

The data are provided as a CSV file with two columns:

> post, <TAB> normalized claim

## Output Data Format

The output must be a CSV format with only one column:

> normalized claim.

## Evaluation Metrics

METEOR [(Banerjee and Lavie, 2005)](https://aclanthology.org/W05-0909.pdf).

## dfkinit2b Submission

**_dfkinit2b_** team is a collaboration between [DFKI](https://dfki.de), [KInIT](https://kinit.sk) and [TU Berlin](https://www.tu.berlin). Our team tested a variety of prompting approaches with different LLMs, fine-tuned adapters, and used an ensemble of methods to find the most representative claim among the generated ones for each post. See our ranking on CodaLab: [https://codalab.lisn.upsaclay.fr/competitions/22801#results](https://codalab.lisn.upsaclay.fr/competitions/22801#results) and the summary with the scores and the best approach for each language below:
![results](https://github.com/user-attachments/assets/33d5887a-b3c5-44b7-91f8-9c0dbae5844a)

## Reproducibility

### Installation

To set up the environment and replicate our experiments, install Python and the required dependencies using:

```bash
pip install -r requirements.txt
```

### Data Preprocessing

Before running the experiments, especially for few-shot prompting, you need to preprocess the data.

To clean the input data and select demonstrations based on cosine similarity, run:

```bash
python -m scripts.cleanup_inputs_re
```

This will create a `./clean_data` directory and generate a list of posts for each language, sorted by cosine similarity. These can be used to select few-shot demonstrations.

To further filter and prepare the data for each language (see [Data](#data)), run: 

```
python src/utils/prepare_data.py --target_lang=all
```

### Prompting Experiments

#### Step 1: Running the Experiments

After preprocessing, you can reproduce the prompting experiments using:

```bash
python -m scripts.prompting_experiments
```

This script will run zero-shot and few-shot prompting experiments across five models, using both translated and English prompts and using filtered and unfiltered data.

### Step 2: Evaluation

To evaluate the outputs of the prompting experiments, open and run the notebook `Final Evaluation.ipynb`.

### Adapter fine-tuning

Adapter fine-tuning with Qwen3 and Gemma3 can be done as follows (example for German with verbose instruction and combined filtered data, replace `model_name` with `unsloth/gemma-3-27b-it` for fine-tuning Gemma3 adapters):

```
python src/models/model_with_adapters.py \
--model_name=unsloth/Qwen3-14B \
--max_seq_length=2048 \
--train_language_code=deu \
--val_language_code=deu \
--input_folder=data/combined_filtered \
--num_epochs=3 \
--learning_rate=2e-4 \
--lr_scheduler=linear \
--instruction_type=verbose_instruction \
--save_adapters_folder=adapters_verbose_instruction_qwen3_2048
```

Next, you will need to load the fine-tuned adapters and run the inference:

```
python src/models/adapter_inference.py \
--model_name=adapters_verbose_instruction_qwen3_2048/Qwen3-14B-deu \
--instruction_type=verbose_instruction \
--target_lang=deu
```

The generated normalized claims will be stored in `outputs/task2_{target_lang}.csv`.

### Ensemble of methods

In order to select the most representative samples with the ensemble methods, you will need to place all the outputs from different models in a folder with the corresponding language id (e.g. `deu/task2_deu.csv`, `deu/task2_deu_v2.csv` etc.) and then run `python src/utils/select_samples_ensemble.py --setting={monolingual|zeroshot|all}`.

## Related Work

Information regarding the task and data can be found in the following paper:

> Megha Sundriyal, Tanmoy Chakraborty, and Preslav Nakov. [From Chaos to Clarity: Claim Normalization to Empower Fact-Checking.](https://aclanthology.org/2023.findings-emnlp.439/) Findings of the Association for Computational Linguistics: EMNLP 2023. 2023. pp. 6594 - 6609.

## Paper citing

If you use the code or information from this repository, please cite our paper.

```bibtex
@misc{anikina2025dfkinit2b,
      title={dfkinit2b at CheckThat! 2025: Leveraging LLMs and Ensemble of Methods for Multilingual Claim Normalization}, 
      author={Tatiana Anikina and Ivan Vykopal and Sebastian Kula Ravi Kiran Chikkala and Natalia Skachkova and Jing Yang and Veronika Solopova and Cera Schmitt and Simon Ostermann},
      year={2025},
}
```
