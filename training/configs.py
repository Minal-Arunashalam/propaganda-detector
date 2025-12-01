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
    model_name = "meta-llama/Llama-3.1-8b-hf"  # Latest free Llama model
    num_labels = 14

    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    test_csv = "data/processed/test.csv"

    max_len = 256
    batch_size = 4  # Smaller batch size for larger model (use 2 for full fine-tuning without LoRA)
    lr = 2e-4  # For LoRA; automatically adjusted to 2e-5 for full fine-tuning if use_lora=False
    epochs = 5
    device = "cuda"  # Will auto-detect in Colab (works with CPU, CUDA, or MPS)

    # LoRA specific parameters
    use_lora = True  # Set to False for full fine-tuning (WARNING: Requires ~28GB+ GPU memory, very slow!)
    # Full fine-tuning may not work on Colab free tier (T4 has ~15GB) - use LoRA instead
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # For Llama architecture
