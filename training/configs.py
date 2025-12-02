class RobertaConfig:
    model_name = "roberta-base"
    num_labels = 14

    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    test_csv = "data/processed/test.csv"

    max_len = 256
    batch_size = 8
    lr = 2e-5
    epochs = 5
    device = "cuda"


class DebertaConfig:
    model_name = "microsoft/deberta-v3-base"
    num_labels = 14

    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    test_csv = "data/processed/test.csv"

    max_len = 256
    batch_size = 6  # Slightly smaller batch size (DeBERTa has slightly more params ~140M vs RoBERTa ~125M)
    lr = 2e-5
    epochs = 5
    device = "cuda"


class LlamaConfig:
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    num_labels = 14

    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    test_csv = "data/processed/test.csv"

    max_len = 128
    batch_size = 1  # Smaller batch size for larger model (use 2 for full fine-tuning without LoRA)
    lr = 2e-4  # For LoRA; automatically adjusted to 2e-5 for full fine-tuning if use_lora=False
    epochs = 2
    device = "cuda"  # Will auto-detect in Colab (works with CPU, CUDA, or MPS)

    # LoRA specific parameters
    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # 4-bit quantization (QLoRA) settings
    load_in_4bit = True
    bnb_4bit_compute_dtype = "bfloat16"
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_use_double_quant = True
