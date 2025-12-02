# Inference script for testing individual examples with trained models
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import json
import os
import sys

from dataset import PropagandaDataset
from model import PropagandaModel
from configs import RobertaConfig, DebertaConfig, LlamaConfig


def get_device(cfg):
    """Auto-detect device"""
    if cfg.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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


def load_roberta_model(epoch=2):
    """Load trained RoBERTa model"""
    cfg = RobertaConfig()
    device = get_device(cfg)
    safe_name = cfg.model_name.replace("/", "_")
    
    checkpoint_path = os.path.join(
        "checkpoints", f"{safe_name}_checkpoint_epoch{epoch}.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = PropagandaModel(cfg.model_name, cfg.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device, cfg


def load_deberta_model(epoch=2):
    """Load trained DeBERTa model"""
    cfg = DebertaConfig()
    device = get_device(cfg)
    safe_name = cfg.model_name.replace("/", "_")
    
    checkpoint_path = os.path.join(
        "checkpoints", f"{safe_name}_checkpoint_epoch{epoch}.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = PropagandaModel(cfg.model_name, cfg.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device, cfg


def load_llama_model(epoch=2, use_lora=True):
    """Load trained Llama model"""
    from peft import PeftModel
    
    cfg = LlamaConfig()
    device = get_device(cfg)
    safe_name = cfg.model_name.replace("/", "_")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if use_lora:
        lora_checkpoint_dir = os.path.join(
            "checkpoints", f"{safe_name}_lora_checkpoint_epoch{epoch}"
        )
        classifier_path = os.path.join(
            "checkpoints", f"{safe_name}_classifier_epoch{epoch}.pt"
        )
        
        if not os.path.exists(lora_checkpoint_dir):
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_checkpoint_dir}")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
        
        base_model = AutoModel.from_pretrained(cfg.model_name)
        lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)
        
        config = AutoConfig.from_pretrained(cfg.model_name)
        model = PropagandaModelWithLoRA(lora_model, config.hidden_size, cfg.num_labels)
        
        classifier_state = torch.load(classifier_path, map_location=device)
        model.classifier.load_state_dict(classifier_state)
    else:
        checkpoint_path = os.path.join(
            "checkpoints", f"{safe_name}_checkpoint_epoch{epoch}.pt"
        )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = PropagandaModel(cfg.model_name, cfg.num_labels)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, device, cfg


def get_label_names(csv_path=None):
    """Get the list of propaganda technique labels"""
    # Try to infer from dataset if CSV path provided
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Get all unique labels by checking all rows
            all_labels_set = set()
            for idx in range(min(100, len(df))):  # Check first 100 rows
                labels = json.loads(df.loc[idx, "labels"])
                # Find which indices are 1
                for i, val in enumerate(labels):
                    if val == 1:
                        all_labels_set.add(i)
            
            # If we found labels, we know the count
            if all_labels_set:
                num_labels = max(all_labels_set) + 1
                # Return placeholder names if we can't determine actual names
                return [f"Technique_{i}" for i in range(num_labels)]
        except:
            pass
    
    # Default label names (14 techniques as per SemEval 2020 Task 11)
    # These should match the order used during training
    label_names = [
        "Loaded Language",
        "Name Calling,Labeling",
        "Repetition",
        "Exaggeration,Minimization",
        "Doubt",
        "Appeal to fear-prejudice",
        "Flag-Waving",
        "Causal Oversimplification",
        "Slogans",
        "Appeal to Authority",
        "Black-and-White Fallacy",
        "Thought-terminating Cliches",
        "Whataboutism,Straw_Men,Red_Herring",
        "Bandwagon,Reductio ad Hitlerum"
    ]
    return label_names


def predict_example(model, tokenizer, text, device, max_len=256, threshold=0.5):
    """Run inference on a single text example"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(logits, tuple):
            logits = logits[1]
    
    probs = torch.sigmoid(logits).cpu().squeeze()
    preds = (probs >= threshold).int()
    
    return probs.numpy(), preds.numpy()


def show_example_results(model_type, example_idx, epoch=2, threshold=0.5, use_lora=True):
    """
    Load a model and show results for a specific example from the test set.
    
    Args:
        model_type: "roberta", "deberta", or "llama"
        example_idx: Index of the example in the test CSV (0-based)
        epoch: Which epoch checkpoint to load (default: 2)
        threshold: Probability threshold for predictions (default: 0.5)
        use_lora: Whether to use LoRA for Llama (default: True)
    """
    # Load model
    print(f"Loading {model_type} model (epoch {epoch})...")
    if model_type.lower() == "roberta":
        model, tokenizer, device, cfg = load_roberta_model(epoch)
    elif model_type.lower() == "deberta":
        model, tokenizer, device, cfg = load_deberta_model(epoch)
    elif model_type.lower() == "llama":
        model, tokenizer, device, cfg = load_llama_model(epoch, use_lora)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model loaded on device: {device}\n")
    
    # Load test dataset
    test_csv = cfg.test_csv
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    df_test = pd.read_csv(test_csv)
    
    if example_idx >= len(df_test):
        raise ValueError(f"Example index {example_idx} is out of range. Test set has {len(df_test)} examples.")
    
    # Get the example
    example = df_test.iloc[example_idx]
    text = example["text"]
    true_labels = json.loads(example["labels"])
    
    # Run prediction
    probs, preds = predict_example(model, tokenizer, text, device, cfg.max_len, threshold)
    
    # Get label names
    label_names = get_label_names(test_csv)
    
    # Display results
    print("=" * 80)
    print(f"Example #{example_idx}")
    print("=" * 80)
    print(f"\nText:")
    print(f'"{text}"')
    print("\n" + "-" * 80)
    print("Propaganda Technique Predictions:")
    print("-" * 80)
    print(f"{'Technique':<50} {'Prob':<8} {'Pred':<6} {'True':<6}")
    print("-" * 80)
    
    for i, label_name in enumerate(label_names):
        prob = probs[i]
        pred = "Yes" if preds[i] == 1 else "No"
        true = "Yes" if true_labels[i] == 1 else "No"
        match = "✓" if preds[i] == true_labels[i] else "✗"
        
        print(f"{label_name:<50} {prob:.4f}   {pred:<6} {true:<6} {match}")
    
    print("-" * 80)
    
    # Summary
    predicted_count = preds.sum()
    true_count = sum(true_labels)
    correct = (preds == true_labels).sum()
    
    print(f"\nSummary:")
    print(f"  Predicted techniques: {predicted_count}")
    print(f"  True techniques: {true_count}")
    print(f"  Correct predictions: {correct}/{len(label_names)}")
    print(f"  Threshold used: {threshold}")
    
    return {
        "text": text,
        "probabilities": probs,
        "predictions": preds,
        "true_labels": true_labels,
        "label_names": label_names
    }


def main():
    """Command-line interface"""
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_type> <example_idx> [epoch] [threshold] [--no-lora]")
        print("  model_type: roberta, deberta, or llama")
        print("  example_idx: Index of example in test set (0-based)")
        print("  epoch: Checkpoint epoch (default: 2)")
        print("  threshold: Prediction threshold (default: 0.5)")
        print("  --no-lora: Use full fine-tuning for Llama (default: use LoRA)")
        print("\nExample:")
        print("  python inference.py roberta 0")
        print("  python inference.py llama 5 3 0.4")
        return
    
    model_type = sys.argv[1].lower()
    example_idx = int(sys.argv[2])
    epoch = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 2
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].replace('.', '').isdigit() else 0.5
    use_lora = "--no-lora" not in sys.argv
    
    show_example_results(model_type, example_idx, epoch, threshold, use_lora)


if __name__ == "__main__":
    main()

