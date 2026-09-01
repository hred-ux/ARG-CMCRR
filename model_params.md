# Model Training Parameters

## 1. Training Hyperparameters

| Parameter | Value | Description |
| --- | ---: | --- |
| batch_size | 64 | Used for train, validation, and test |
| use_gpu | True | Uses cuda:0 |
| lr | 0.001 | Adam learning rate |
| epochs | 500 | Trains for 500 epochs by default; epoch indices are 0..499 |
| max_seq_len | 400 | Default for long sequences; short-sequence experiments use 40 |
| DataLoader shuffle | train=True; val/test=False | Data loading order |

## 2. Model Construction Parameters

The training entry point calls:

    RNNClassifier(
        vocab_size=22,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=18,
    )

| Model Parameter | Value |
| --- | ---: |
| vocab_size | 22 |
| embedding_dim | 128 |
| hidden_dim | 128 |
| num_classes | 18/7 |
| Default constructor dropout | 0.5 |

Fixed dropout values:

| Location | Dropout |
| --- | ---: |
| embedding | 0.2 |
| Each convolution block | 0.3 |
| Top-level Multi-Head Self-Attention | 0.5 |
| GatedFusion | 0.3 |
| Classifier | 0.2 |
| Internal ProteinBERT attention | 0.1 |

## 3. Optimizer

    torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

Adam parameters use PyTorch defaults:

| Parameter | Value |
| --- | ---: |
| betas | (0.9, 0.999) |
| eps | 1e-8 |
| amsgrad | False |


## 4. Loss Function

    nn.CrossEntropyLoss()

- class_weight is not used
- label_smoothing is not used
- No other regularization loss is used

## 5. Evaluation Metrics

The following metrics are calculated for the training and validation sets at each epoch:

- multiclass Accuracy
- weighted Precision
- weighted Recall
- weighted F1

The following additional metric is calculated for the test set:

- macro F1

