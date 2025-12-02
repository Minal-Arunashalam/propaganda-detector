## Propaganda Detector

This project is a **multi‑label text classification** system for detecting propaganda techniques in text.  
It provides training and evaluation pipelines for three families of transformer models:

- **RoBERTa** (`roberta-base`)
- **DeBERTa** (`microsoft/deberta-v3-base`)
- **Llama 3.1** (`meta-llama/Llama-3.1-8b-hf`), with **LoRA parameter‑efficient fine‑tuning**

The code is written in **PyTorch** and built on top of **Hugging Face Transformers**, with additional tooling from **PEFT**, **Accelerate**, **bitsandbytes**, and **scikit‑learn**.

---

## Project Structure

- `training/train.py` – main training script; trains RoBERTa, DeBERTa, or Llama models.
- `training/eval.py` – evaluation script; loads checkpoints and reports precision/recall/F1 at multiple thresholds.
- `training/dataset.py` – `PropagandaDataset` class that reads processed CSV files and tokenizes text.
- `training/configs.py` – configuration classes for RoBERTa, DeBERTa, and Llama (paths, hyperparameters, LoRA settings).
- `training/model.py` – model definition used by training/eval (encoder + classification head).
- `data/processed/*.csv` – expected location of preprocessed train/val/test splits.
- `checkpoints/` – where model checkpoints and LoRA adapters are saved.

---

## Tools and Libraries Used

- **PyTorch (`torch`)**

  - Core deep learning framework.
  - Defines models (`nn.Module`), loss functions (e.g., `BCEWithLogitsLoss`), and optimizers (`AdamW`).
  - Handles GPU/CPU/MPS device placement and batched training via `DataLoader`.

- **Hugging Face Transformers (`transformers`)**

  - `AutoTokenizer`, `AutoModel`, and model configs for RoBERTa, DeBERTa, and Llama 3.1.
  - Provides pretrained encoders that are fine‑tuned for multi‑label classification.

- **PEFT (`peft`)**

  - Used for **LoRA (Low‑Rank Adaptation)** fine‑tuning of Llama 3.1.
  - Allows training only a small number of additional parameters on top of a frozen base model, making large‑model fine‑tuning feasible on limited hardware.

- **bitsandbytes / Accelerate**

  - Used for efficient training and memory‑saving setups with large language models (e.g., 8‑bit or 4‑bit quantization, distributed/accelerated training).

- **Pandas (`pandas`)**

  - Loads and manipulates CSV datasets (`train.csv`, `val.csv`, `test.csv`).

- **Scikit‑learn (`scikit-learn`)**

  - Computes evaluation metrics:
    - micro/macro **precision**
    - micro/macro **recall**
    - micro/macro **F1‑score**

- **Standard Python libraries**
  - `json`, `csv`, `os`, `datetime`, etc., for logging, checkpoint management, and configuration.

---

## Techniques and Modeling Approach

- **Multi‑label classification**

  - Each text can contain **multiple propaganda techniques** simultaneously.
  - The model outputs one logit per label (14 total), and **`BCEWithLogitsLoss`** is used during training.

- **Transformer encoders**

  - RoBERTa and DeBERTa use the standard encoder‑based architecture from Hugging Face.
  - Llama 3.1 uses an LLM encoder; a linear classification head is applied on top of the `[CLS]`/first token representation.

- **LoRA fine‑tuning for Llama 3.1**

  - When `use_lora=True` in `LlamaConfig`, LoRA adapters are inserted into attention modules (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj`).
  - Only these adapter weights (and the classifier head) are trained, dramatically reducing memory and compute requirements.
  - When `use_lora=False`, the code performs **full fine‑tuning**, which requires ~28GB+ GPU memory.

- **Device auto‑detection**

  - `get_device` in `train.py` / `eval.py` automatically picks:
    - `"cuda"` if a GPU is available,
    - `"mps"` on Apple Silicon,
    - otherwise `"cpu"`.

- **Threshold search during evaluation**

  - Evaluation runs the model once on the test set, collects logits and labels, and then tries thresholds from 0.1 to 0.9.
  - For each threshold, micro/macro precision, recall, and F1 are computed.
  - The best threshold by micro F1 is reported.

- **Logging and checkpointing**
  - After each epoch, training and validation loss are logged to both:
    - a CSV file (`*_training_results.csv`)
    - and a JSON file (`*_training_results.json`).
  - RoBERTa/DeBERTa save **full model checkpoints**.
  - Llama with LoRA saves:
    - LoRA adapter directories per epoch, and
    - a separate classifier head checkpoint.

---

## Installation

1. **Clone the repository** and move into it:

```bash
git clone <your-repo-url>.git
cd propaganda-detector
```

2. **Create and activate a virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Data Format

The training, validation, and test splits are expected under `data/processed/`:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

Each CSV should contain at least:

- **`text`**: the input sentence or document.
- **`labels`**: a JSON‑encoded list/array of 0/1 values, one per propaganda technique (length 14 by default).

The `PropagandaDataset` in `training/dataset.py`:

- loads the CSV with pandas,
- tokenizes `text` using a Hugging Face tokenizer,
- truncates/pads to `max_len`,
- returns `input_ids`, `attention_mask`, and `labels` tensors.

---

## Configuration

Model and training hyperparameters are defined in `training/configs.py`:

- **`RobertaConfig`**

  - `model_name = "roberta-base"`
  - `max_len = 256`, `batch_size = 8`, `lr = 2e-5`, `epochs = 5`
  - CSV paths under `data/processed/`

- **`DebertaConfig`**

  - `model_name = "microsoft/deberta-v3-base"`
  - `batch_size = 6` (slightly smaller than with roberta due to more parameters)
  - Other settings similar to `RobertaConfig`

- **`LlamaConfig`**
  - `model_name = "meta-llama/Llama-3.1-8b-hf"`
  - `batch_size = 4` by default
  - `lr = 2e-4` for LoRA
  - LoRA‑specific settings: `use_lora`, `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`

These classes can be modified to point to different models, datasets, or hyperparameters.

---

## Training

From the `training/` directory (or with an appropriate `PYTHONPATH`), you can start training by specifying which backbone to use:

- **RoBERTa**:

```bash
python training/train.py roberta
```

- **DeBERTa**:

```bash
python training/train.py deberta
```

- **Llama 3.1 with LoRA**:

```bash
python training/train.py llama
```

During Llama training, `LlamaConfig.use_lora` controls whether LoRA or full fine‑tuning is used.

Checkpoints and logs will be written under:

- `checkpoints/`
- `<model_name>_training_results.csv`
- `<model_name>_training_results.json`

---

## Evaluation

Use `training/eval.py` to evaluate saved checkpoints and search for the best decision threshold.

Example commands:

- **RoBERTa** (epoch 2):

```bash
python training/eval.py roberta 2
```

- **DeBERTa** (epoch 3):

```bash
python training/eval.py deberta 3
```

- **Llama 3.1 with LoRA** (epoch 2, default LoRA):

```bash
python training/eval.py llama 2
```

- **Llama 3.1 full fine‑tuning** (no LoRA):

```bash
python training/eval.py llama 2 --no-lora
```

The script:

- loads the specified checkpoint(s),
- runs inference over the test set to collect logits and labels,
- sweeps thresholds from 0.1 to 0.9,
- prints micro/macro precision, recall, and F1,
- and highlights the threshold with the best micro F1.

---

## Hardware Notes

- RoBERTa and DeBERTa run on a single modern GPU with 8–12GB of VRAM.
- Llama 3.1 (8B) is significantly larger:
  - **LoRA fine‑tuning** makes this feasible on smaller GPUs.
  - **Full fine‑tuning** requires GPUs with ~28GB+ memory.

---

## License and Attribution

- The Llama 3.1 model weights are distributed by Meta and subject to the **Meta Llama license**; make sure to review and comply with it.
- Pretrained RoBERTa and DeBERTa models are provided by Hugging Face / respective authors under their own licenses.
- This repository’s code can be reused or extended according to the license you choose to apply here.

If you use this project in academic work, please consider citing the original propaganda detection dataset and the underlying model papers (RoBERTa, DeBERTa, Llama, LoRA, etc.).
