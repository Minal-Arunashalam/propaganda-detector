class Config:
    model_name = "roberta-base"
    num_labels = 14

    train_csv = "processed/train.csv"
    val_csv = "processed/val.csv"
    test_csv = "processed/test.csv"

    max_len = 256
    batch_size = 8
    lr = 2e-5
    epochs = 5
    device = "cuda"
