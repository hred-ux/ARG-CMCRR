# ARG-CMCRR Model Training and Inference Settings

## 1. Project Goals and Task Formats

ARG-CMCRR is used to perform the following tasks from amino acid or nucleotide sequences:

- Antibiotic resistance gene (ARG) identification;
- ARG functional-category classification;
- ARG mechanism-of-action classification;
- Risk-level prediction for full-length amino acid sequences.

The repository supports four basic input combinations. Full-length amino acid input also provides an extended execution path with risk prediction:

| Input type | Length mode | Model/pipeline | Output |
|---|---|---|---|
| Amino acid | Short sequence | `argcmcrr_ssaa.py` | ARG class, mechanism of action |
| Nucleotide | Short sequence | `argcmcrr_ssnt.py` | ARG class, mechanism of action |
| Amino acid | Full-length sequence | `argcmcrr_lsaa.py` | ARG class, mechanism of action |
| Amino acid | Full-length sequence + risk | `argcmcrr_lsaa_with_risk.py` | ARG class, mechanism of action, risk level, BLAST risk |
| Nucleotide | Full-length sequence | `argcmcrr_lsnt.py` | ARG class, mechanism of action |

Risk prediction is enabled only in the full-length amino acid mode. The top-level entry point, `ARG-CMCRR.py`, automatically routes to the appropriate script according to `--type`, `--length`, and `--risk`.

## 2. Summary of Training Configuration

The following table lists the training configuration used in this project. `batch_size`, learning rate, number of epochs, optimizer, and loss function are the training settings explicitly recorded by the project.

| Parameter | Recorded value | Description |
|---|---:|---|
| Device | GPU, `cuda:0` | `use_gpu=True` |
| Batch size | `64` | 64 is used for training, validation, and testing |
| Initial learning rate | `0.001` | Adam learning rate |
| Optimizer | Adam | See Section 7 |
| Number of epochs | `500` | Epoch indices are `0..499` |
| Maximum length for long sequences | `400` | Default value recorded in the training parameters |
| Short-sequence length | `40` | Used in short-sequence experiments |
| Training-set shuffle | `True` | Recorded `DataLoader` setting |
| Validation-set shuffle | `False` | Recorded `DataLoader` setting |
| Test-set shuffle | `False` | Recorded `DataLoader` setting |
| Loss function | `CrossEntropyLoss` | No class weights or label smoothing are used |
| Weight decay | `1e-4` | Adam parameter |

## 3. Input Data and Preprocessing

### 4.1 Amino Acid Sequences

Sequences are read using `FastaBatchedDataset.from_file()` from `fair-esm`.

| Mode | Maximum length | Filtering condition | Processing |
|---|---:|---|---|
| Short amino acid `ssaa` | `40` | No additional filtering in `utils.py` | Sequences longer than 40 are truncated; shorter sequences are padded with zeros to length 40 |
| Full-length amino acid `lsaa` | `400` | Raw sequences shorter than 50 are skipped | Sequences longer than 400 are truncated; shorter sequences are padded with zeros to length 400 |

The README recommends a range of 30–50 aa for short amino acid sequences; the code actually normalizes all inputs to a length of 40. The full-length amino acid code requires an original length of at least 50 aa, but the actual model input retains at most the first 400 residues.

### 4.2 Nucleotide Sequences

Nucleotide input is first translated in six frames using `Bio.Seq.Seq`: three reading frames on the forward strand plus three reading frames on the reverse-complement strand.

The selection rules in the code are:

1. If a translated frame does not contain `*`, the first frame satisfying this condition is selected;
2. Otherwise, the translation result is split at `*`, and the longest contiguous segment is selected;
3. The full-length nucleotide mode requires the longest translated segment to be at least 50 aa;
4. The short nucleotide mode requires the longest translated segment to be at least 30 aa;
5. The translated amino acid sequence is then processed using a maximum length of either 400 or 40.

| Mode | Maximum translated length | Minimum valid translated length |
|---|---:|---:|
| Short nucleotide `ssnt` | `40` aa | `30` aa |
| Full-length nucleotide `lsnt` | `400` aa | `50` aa |

### 4.3 Sequence Encoding

Integer token encoding is used. The vocabulary size is 22:

| Token | ID | Token | ID |
|---|---:|---|---:|
| `<pad>` | 0 | `N` | 12 |
| `A` | 1 | `P` | 13 |
| `C` | 2 | `Q` | 14 |
| `D` | 3 | `R` | 15 |
| `E` | 4 | `S` | 16 |
| `F` | 5 | `T` | 17 |
| `G` | 6 | `V` | 18 |
| `H` | 7 | `W` | 19 |
| `I` | 8 | `Y` | 20 |
| `K` | 9 | `X` | 21 |
| `L` | 10 | Unknown character | 0 |
| `M` | 11 |  |  |

Encoding details:

- The ID of `<pad>` is 0;
- Sequences shorter than the maximum length are right-padded with zeros;
- Sequences longer than the maximum length are truncated directly;
- Amino acid characters not in the dictionary are mapped to 0, so they share the same ID as padding;
- `actual_len` stores the effective length after truncation and is passed to `pack_padded_sequence`.

## 5. Classification Labels

### 5.1 ARG Classes: 18 Classes

```text
0  beta-lactam
1  bacitracin
2  multidrug
3  MLS
4  aminoglycoside
5  polymyxin
6  fosfomycin
7  quinolone
8  chloramphenicol
9  tetracycline
10 glycopeptide
11 peptide
12 sulfonamide
13 trimethoprim
14 novobiocin
15 rifamycin
16 nonarg
17 others
```

Samples predicted as `nonarg` by the code are labeled as non-ARG. The remaining 17 classes are treated as ARGs and passed to mechanism-of-action prediction.

### 5.2 Mechanisms of Action: 6 Classes

```text
0  antibiotic inactivation
1  antibiotic target alteration
2  antibiotic efflux
3  antibiotic target replacement
4  antibiotic target protection
5  others
```

The mechanism-of-action model is not applied directly to all inputs. It is applied only to samples predicted as ARGs in the first stage.

## Model Architecture

The actual inference scripts use `Classifier` from `classifier.py`, with the following constructor parameters:

```python
Classifier(
    vocab_size=22,
    embedding_dim=128,
    hidden_dim=128,
    num_classes=18,  # ARG classification model
)
```

The mechanism-of-action model has the same architecture, with `num_classes` changed to 6.

### 6.1 Backbone Network

```text
Integer sequence [B, L]
    ↓
Embedding(22 → 128)
    ↓
Sinusoidal positional encoding (max_len=500)
    ↓
Dropout(0.2)
    ↓
Two parallel Conv1d branches:
  - kernel=3, 128 → 64
  - kernel=5, 128 → 64
Each branch: BatchNorm1d → GELU → Dropout(0.3)
    ↓ Concatenation
Feature dimension 128
    ↓
Bidirectional GRU: 2 layers, hidden_size=128
Output dimension 128 × 2 = 256
    ↓
4-head Multi-Head Self-Attention
    ↓
Concatenation with the ProteinBERT branch
    ↓
Fusion layer: 512 → 256
LayerNorm → GELU → Dropout(0.3)
    ↓
Average of global max pooling and global average pooling
    ↓
Classification head: 256 → 128 → num_classes
```

### 6.2 Detailed Module Parameters

| Module | Parameters |
|---|---|
| Token embedding | `nn.Embedding(22, 128)` |
| Positional encoding | Sinusoidal positional encoding, `max_len=500` |
| Embedding dropout | `0.2` |
| Convolution branch 1 | `Conv1d(128, 64, kernel_size=3, padding=1)` |
| Convolution branch 2 | `Conv1d(128, 64, kernel_size=5, padding=2)` |
| Convolution-branch normalization | `BatchNorm1d(64)` for each branch |
| Convolution-branch activation | `GELU` |
| Convolution-branch dropout | `0.3` |
| GRU | 2 layers, bidirectional, `input_size=128`, `hidden_size=128`, `batch_first=True` |
| GRU dropout | Not set; PyTorch default is `0` |
| Top-level self-attention | Hidden dimension `256`, 4 heads, dropout `0.5` |
| QKV projection | `Linear(256, 768)` |
| ProteinBERT branch | `dim=128`, `depth=2`, 4 heads |
| ProteinBERT local convolution | Narrow kernel `9`; wide kernel `9`; dilation `5` |
| Relative-position range | `max_rel_pos=50` |
| ProteinBERT attention dropout | `0.1` |
| ProteinBERT output | `Linear(128, 128)` after LayerNorm |
| BERT adapter | `Linear(128, 256)` |
| Fusion layer | `Linear(512, 256)` → LayerNorm → GELU → Dropout `0.3` |
| Classification head | `Linear(256, 128)` → GELU → Dropout `0.2` → `Linear(128, num_classes)` |

### 6.3 Sequence Fusion and Pooling

The backbone branch produces 256-dimensional temporal features. After passing through the adapter, the ProteinBERT branch also produces 256-dimensional features. The two are concatenated along the last dimension to form a 512-dimensional representation, which then passes through the 512→256 fusion layer.

The final temporal features are aggregated as follows:

```text
global_max_pool = max(fused, dim=1)
global_avg_pool = mean(fused, dim=1)
final_features = (global_max_pool + global_avg_pool) / 2
```

The code does not pass a padding mask when calling the top-level attention layer. In addition, global average pooling also averages over the padded positions produced by `pad_packed_sequence`.

### 6.4 Status of the `models_gated` Directory

The repository also contains `models_gated/classifier_gated.py`, which defines `RNNClassifier` and `GatedFusion`. This version adds gated fusion.

## 6. Optimizer, Loss Function, and Evaluation Metrics

### 7.1 Optimizer

The optimizer is configured as follows:

```python
torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)
```

Adam uses the PyTorch default values:

| Parameter | Value |
|---|---:|
| `betas` | `(0.9, 0.999)` |
| `eps` | `1e-8` |
| `amsgrad` | `False` |
| `weight_decay` | `1e-4` |

### 7.2 Loss Function

```python
nn.CrossEntropyLoss()
```

### 7.3 Evaluation Metrics

The epoch-level metrics for the training and validation sets are:

- Multiclass accuracy;
- Weighted precision;
- Weighted recall;
- Weighted F1 score.

The test set additionally records:

- Macro F1 score.

## 7. Inference Batch Sizes by Mode

| Mode | Classification/mechanism inference batch size | Risk feature extraction | XGBoost risk prediction |
|---|---:|---:|---:|
| Short amino acid | `64` | Not applicable | Not applicable |
| Short nucleotide | `64` | Not applicable | Not applicable |
| Full-length amino acid | `64` | Not applicable | Not applicable |
| Full-length nucleotide | `64` | Not applicable | Not applicable |
| Full-length amino acid + risk | `32` | `1` | `16` |

The full-length amino acid risk script also defines a default batch size of 32 for `blast_in_batches`. However, the main pipeline actually uses `blast_all_at_once` and calls BLAST once for all ARG sequences.

## 8. Additional Risk-Prediction Pipeline

Risk prediction is a separate three-stage pipeline:

```text
Full-length amino acid input
    ↓
ESM-2 650M CLS embedding extraction
    ↓
XGBoost classifier predicts risk level
    ↓
BLASTP best-hit search against the ARG-Risk-21 and ARG-Risk-22 databases
    ↓
Output risk_label, risk_probability, blast_21_risk, blast_22_risk
```

### 9.1 ESM-2 Settings

| Parameter | Setting |
|---|---|
| Model | `facebook/esm2_t33_650M_UR50D` |
| Local directory | `esm2_t33_650M/` |
| Loading method | `transformers.AutoTokenizer` + `AutoModel` |
| Tokenizer | `do_lower_case=False` |
| Input processing | Amino acid characters in each sequence are joined with spaces |
| Padding | `padding=True` |
| Inference mode | `.eval()` + `torch.no_grad()` |
| Precision | The model is converted to `.half()` and run under CUDA autocast |
| Features | `outputs.last_hidden_state[:, 0, :]`, i.e. the vector at the CLS position |
| Feature-extraction batch size | `1` |

### 9.2 XGBoost Settings

The repository provides:

```text
risk_modelfile/xgb_esm2_650m.json
risk_modelfile/label_encoder.pkl
```

The code loads them as follows:

```python
clf = XGBClassifier()
clf.load_model("risk_modelfile/xgb_esm2_650m.json")
le = joblib.load("risk_modelfile/label_encoder.pkl")
```

### 9.3 BLASTP Settings

The risk script calls `blast/bin/blastp` in the repository directory. The main parameters are:

| Parameter | Value |
|---|---:|
| `-outfmt` | `6 qseqid sseqid bitscore evalue` |
| `-evalue` | `1e-3` |
| `-max_target_seqs` | `1` |
| `-max_hsps` | `1` |
| `-num_threads` | `16` (main risk pipeline) |
| Database files | `blastfile/ARG-Risk-21.fasta`, `blastfile/ARG-Risk-22.fasta` |

## Software Environment and External Dependencies

The repository's `environment.yml` specifies:

| Dependency | Version |
|---|---:|
| Python | `3.9` |
| PyTorch | `2.0.0`, CUDA 11.8 wheel |
| torchvision | `0.15.1`, CUDA 11.8 wheel |
| torchaudio | `2.0.1`, CUDA 11.8 wheel |
| tqdm | `4.65.0` |
| bio | `1.6.2` |
| biopython | `1.83` |
| einops | `0.8.0` |
| fair-esm | `2.0.0` |
| numpy | `1.24.3` |
| pandas | `2.0.2` |
| scikit-learn | `1.2.2` |
| transformers | `4.46.3` |
| xgboost | `1.7.5` |
| joblib | `1.2.0` |

Environment-creation commands:

```bash
conda env create -n ARG-CMCRR -f environment.yml
conda activate ARG-CMCRR
```

The README requires:

- An NVIDIA GPU;
- CUDA ≥ 11.x is recommended;
- At least 8 GB of GPU memory is recommended for long-sequence prediction;
- CPU-only execution is not supported.

The following must also be prepared separately:

1. The ESM-2 650M model, placed in `esm2_t33_650M/`;
2. NCBI BLAST+, with its `blastp` executable placed in `blast/bin/`.

## 10. Published Weights and Resource Files

Neural-network classification weights are located at:

```text
classptFile/aa_short_class.pt
classptFile/aa_long_class.pt
classptFile/nt_short_class.pt
classptFile/nt_long_class.pt

mechptFile/aa_short_mechanism.pt
mechptFile/aa_long_mechanism.pt
mechptFile/nt_short_mechanism.pt
mechptFile/nt_long_mechanism.pt
```

Weights are loaded using `torch.load(..., map_location=torch.device("cuda"))`, after which the model is moved to CUDA. Except for the risk script, which explicitly calls `.eval()`, the other scripts call `model.eval()` inside their prediction functions and perform the forward pass within a `torch.no_grad()` context.

## 11. Execution Commands

### 12.1 Full-Length Amino Acid + Risk Prediction

```bash
python ARG-CMCRR.py \
  --input fastafile/aa_long_test.fasta \
  --type aa \
  --length l \
  --outfile results \
  --risk True
```

### 12.2 Full-Length Amino Acid, Classification Only

```bash
python ARG-CMCRR.py \
  --input input.fasta \
  --type aa \
  --length l \
  --outfile results \
  --risk False
```

### 12.3 Full-Length Nucleotide

```bash
python ARG-CMCRR.py \
  --input input.fasta \
  --type nt \
  --length l \
  --outfile results
```

### 12.4 Short Amino Acid

```bash
python ARG-CMCRR.py \
  --input input.fasta \
  --type aa \
  --length s \
  --outfile results
```

### 12.5 Short Nucleotide

```bash
python ARG-CMCRR.py \
  --input input.fasta \
  --type nt \
  --length s \
  --outfile results
```

## 12. Output Fields

The standard classification mode usually outputs:

```text
header
sequence
Arg/non-Arg
pred_class
probability_class
pred_mech
probability_mech
```

The risk mode adds the following fields:

```text
risk_label
risk_probability
blast_21_risk
blast_22_risk
```

For sequences predicted as `non-ARG`, the mechanism-of-action fields are empty. In risk mode, the risk fields for non-ARG sequences are also empty.

## 13. Related File Information

| File | Purpose |
|---|---|
| `README.md` | Project goals, input modes, installation, and execution instructions |
| `classifier.py` | The main classifier architecture currently used |
| `proteinbert_simply.py` | ProteinBERT branch and relative-position attention |
| `utils.py` | FASTA reading, six-frame translation, encoding, truncation, and padding |
| `argcmcrr_ssaa.py` | Short amino acid inference configuration |
| `argcmcrr_ssnt.py` | Short nucleotide inference configuration |
| `argcmcrr_lsaa.py` | Full-length amino acid inference configuration |
| `argcmcrr_lsaa_with_risk.py` | Full-length amino acid risk-inference configuration |
| `argcmcrr_lsnt.py` | Full-length nucleotide inference configuration |
| `environment.yml` | Versions of Python, PyTorch, and third-party dependencies |
| `models_gated/classifier_gated.py` | Gated-model variant not called by the current inference entry point |
