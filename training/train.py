# ties everything together and is the file to run
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
import json
import csv
import os
from datetime import datetime

from dataset import PropagandaDataset
from model import PropagandaModel
from configs import RobertaConfig, DebertaConfig, LlamaConfig


def get_device(cfg):
    """Auto-detect device, works in Colab"""
    if cfg.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


def log_results(safe_name, epoch, train_loss, val_loss, use_lora=False):
    """Log training results to both CSV and JSON files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = f"{safe_name}_lora" if use_lora else safe_name

    # CSV logging
    csv_file = f"{model_prefix}_training_results.csv"
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "timestamp"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", timestamp])

    # JSON logging (append to list)
    json_file = f"{model_prefix}_training_results.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    results.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": timestamp,
        }
    )

    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #progress print every 10 steps
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{len(loader)} - loss: {loss.item():.4f}")

    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()  # eval mode
    total_loss = 0

    with torch.no_grad():  # no gradients for validation
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            total_loss += loss.item()

    return total_loss / len(loader)


def train_roberta():
    cfg = RobertaConfig()
    safe_name = cfg.model_name.replace("/", "_")
    device = get_device(cfg)

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = PropagandaDataset(cfg.train_csv, tokenizer, cfg.max_len)
    val_ds = PropagandaDataset(cfg.val_csv, tokenizer, cfg.max_len)

    # Compute class-wise pos_weight for BCEWithLogitsLoss to handle class imbalance
    all_labels = []
    for i in range(len(train_ds)):
        labels_i = train_ds[i]["labels"]
        # Ensure we always have a float32 tensor
        if isinstance(labels_i, torch.Tensor):
            all_labels.append(labels_i.float())
        else:
            all_labels.append(torch.tensor(labels_i, dtype=torch.float32))
    all_labels_tensor = torch.stack(all_labels, dim=0)  # [N, num_labels]
    pos_counts = all_labels_tensor.sum(dim=0)
    neg_counts = all_labels_tensor.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = PropagandaModel(cfg.model_name, cfg.num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"Training RoBERTa model: {cfg.model_name}")
    print(
        f"Checkpoints will be saved to: {checkpoint_dir}/{safe_name}_checkpoint_epoch*.pt"
    )
    print(f"Results will be saved to: {safe_name}_training_results.csv and .json")

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Log results to files
        log_results(safe_name, epoch + 1, train_loss, val_loss, use_lora=False)

        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{safe_name}_checkpoint_epoch{epoch+1}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    print(f"\nRoBERTa training completed!")
    print(f"All checkpoints saved in: {checkpoint_dir}/")
    print(f"Training results saved to: {safe_name}_training_results.csv and .json")


def train_deberta():
    cfg = DebertaConfig()
    safe_name = cfg.model_name.replace("/", "_")
    device = get_device(cfg)

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = PropagandaDataset(cfg.train_csv, tokenizer, cfg.max_len)
    val_ds = PropagandaDataset(cfg.val_csv, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = PropagandaModel(cfg.model_name, cfg.num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"Training DeBERTa model: {cfg.model_name}")
    print(
        f"Checkpoints will be saved to: {checkpoint_dir}/{safe_name}_checkpoint_epoch*.pt"
    )
    print(f"Results will be saved to: {safe_name}_training_results.csv and .json")

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Log results to files
        log_results(safe_name, epoch + 1, train_loss, val_loss, use_lora=False)

        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{safe_name}_checkpoint_epoch{epoch+1}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    print(f"\nDeBERTa training completed!")
    print(f"All checkpoints saved in: {checkpoint_dir}/")
    print(f"Training results saved to: {safe_name}_training_results.csv and .json")


def train_llama():
    from peft import LoraConfig, get_peft_model, TaskType

    cfg = LlamaConfig()
    safe_name = cfg.model_name.replace("/", "_")
    device = get_device(cfg)

    print(f"Using device: {device}")

    # Warn about full fine-tuning resource requirements
    if not cfg.use_lora:
        print("\n" + "=" * 70)
        print("WARNING: Full fine-tuning of Llama-3.1-8B is VERY resource intensive!")
        print("This requires:")
        print("  - ~28GB+ GPU memory (even with batch_size=2)")
        print("  - Multiple hours per epoch on most hardware")
        print("  - May not work on Colab free tier (T4 GPU has ~15GB)")
        print("\nRecommendation: Use LoRA fine-tuning (set use_lora=True in config)")
        print("=" * 70 + "\n")

        # Adjust learning rate for full fine-tuning
        if cfg.lr == 2e-4:  # If still using LoRA LR, adjust it
            cfg.lr = 2e-5
            print(f"Adjusted learning rate to {cfg.lr} for full fine-tuning")

        # Further reduce batch size if needed
        if cfg.batch_size > 2:
            cfg.batch_size = 2
            print(f"Reduced batch size to {cfg.batch_size} for full fine-tuning")

    # Llama tokenizer requires padding token
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = PropagandaDataset(cfg.train_csv, tokenizer, cfg.max_len)
    val_ds = PropagandaDataset(cfg.val_csv, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create base model and apply LoRA to encoder
    if cfg.use_lora:
        print(f"Training Llama with LoRA: {cfg.model_name}")
        # Load the base encoder (optionally in 4-bit for QLoRA)
        if getattr(cfg, "load_in_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=getattr(cfg, "bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_type=getattr(cfg, "bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModel.from_pretrained(
                cfg.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            base_model = AutoModel.from_pretrained(cfg.model_name).to(device)

        # Apply LoRA to the encoder
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
        )
        lora_model = get_peft_model(base_model, peft_config)
        lora_model.print_trainable_parameters()

        # Now create the full model with LoRA-adapted encoder
        config = AutoConfig.from_pretrained(cfg.model_name)
        # Pass class imbalance weights into the model so the loss can use them
        model = PropagandaModelWithLoRA(
            lora_model, config.hidden_size, cfg.num_labels, pos_weight=pos_weight
        )
        # For non-quantized models, move everything to the chosen device
        if not getattr(cfg, "load_in_4bit", False):
            model = model.to(device)

        # Only optimize trainable parameters (LoRA parameters + classifier)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    else:
        print(f"Training Llama with FULL fine-tuning: {cfg.model_name}")
        model = PropagandaModel(cfg.model_name, cfg.num_labels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    results_prefix = f"{safe_name}_lora" if cfg.use_lora else safe_name
    if cfg.use_lora:
        print(
            f"LoRA checkpoints will be saved to: {checkpoint_dir}/{safe_name}_lora_checkpoint_epoch*/"
        )
        print(
            f"Classifier will be saved to: {checkpoint_dir}/{safe_name}_classifier_epoch*.pt"
        )
    else:
        print(
            f"Checkpoints will be saved to: {checkpoint_dir}/{safe_name}_checkpoint_epoch*.pt"
        )
    print(f"Results will be saved to: {results_prefix}_training_results.csv and .json")

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Log results to files
        log_results(safe_name, epoch + 1, train_loss, val_loss, use_lora=cfg.use_lora)

        # Save LoRA adapters if using LoRA, otherwise save full model
        if cfg.use_lora:
            # Save LoRA adapters (creates a directory)
            lora_checkpoint_path = os.path.join(
                checkpoint_dir, f"{safe_name}_lora_checkpoint_epoch{epoch+1}"
            )
            model.encoder.save_pretrained(lora_checkpoint_path)
            # Also save classifier
            classifier_path = os.path.join(
                checkpoint_dir, f"{safe_name}_classifier_epoch{epoch+1}.pt"
            )
            torch.save(model.classifier.state_dict(), classifier_path)
            print(f"  ✓ Saved LoRA checkpoint: {lora_checkpoint_path}/")
            print(f"  ✓ Saved classifier: {classifier_path}")
        else:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{safe_name}_checkpoint_epoch{epoch+1}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    print(f"\nLlama training completed!")
    print(f"All checkpoints saved in: {checkpoint_dir}/")
    print(f"Training results saved to: {results_prefix}_training_results.csv and .json")


# Helper class for Llama with LoRA
class PropagandaModelWithLoRA(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels, pos_weight=None):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Optional class-imbalance weights for BCEWithLogitsLoss
        if pos_weight is not None:
            # Register as buffer so it moves with the model between devices
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get encoder output and ensure classifier is on the same device
        hidden = outputs.last_hidden_state
        device = hidden.device

        # If classifier is not on this device yet, move it once
        if self.classifier.weight.device != device:
            self.classifier.to(device)

        # CLS token representation
        cls = hidden[:, 0, :]  # [batch, hidden_size]

        # Make sure dtype matches classifier weights (fixes Half vs Float error)
        if cls.dtype != self.classifier.weight.dtype:
            cls = cls.to(self.classifier.weight.dtype)

        logits = self.classifier(cls)

        if labels is not None:
            # Ensure labels and pos_weight live on the same device/dtype as logits
            labels = labels.to(device).to(logits.dtype)

            if getattr(self, "pos_weight", None) is not None:
                pw = self.pos_weight.to(logits.device).to(logits.dtype)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                loss_fct = nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels)
            return loss, logits

        return logits


def main():
    import sys

    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type == "roberta":
            train_roberta()
        elif model_type == "deberta":
            train_deberta()
        elif model_type == "llama":
            train_llama()
        else:
            print(f"Unknown model type: {model_type}")
            print("Usage: python train.py [roberta|deberta|llama]")
    else:
        print("Please specify a model type:")
        print("  python train.py roberta")
        print("  python train.py deberta")
        print("  python train.py llama")


if __name__ == "__main__":
    main()
