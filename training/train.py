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
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss, logits = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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

    if not cfg.use_lora:
        if cfg.lr == 2e-4:
            cfg.lr = 2e-5

        if cfg.batch_size > 2:
            cfg.batch_size = 2

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
        model = PropagandaModelWithLoRA(
            lora_model, config.hidden_size, cfg.num_labels
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
    def __init__(self, encoder, hidden_size, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

        if labels is not None:
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
    else:
        print("Specify model type: roberta, deberta, or llama")


if __name__ == "__main__":
    main()
