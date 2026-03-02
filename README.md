# ARG-CMCRR

## ARG_CMCRR: Deep Learning-Based Classification and Risk Ranking of Antibiotic Resistance Genes

ARG_CMCRR is a GPU-accelerated deep learning framework for the identification, functional classification, and risk ranking of antibiotic resistance genes (ARGs) from nucleotide or amino acid sequences.

The framework supports both full-length sequences and short sequencing reads, enabling its application to assembled genomes, plasmids, and metagenomic datasets.

---

## 1. Overview

ARG_CMCRR provides:

* **ARG identification and functional classification**
* **Risk level prediction (for full-length amino acid sequences)**
* Support for:

  * Long amino acid sequences
  * Long nucleotide sequences
  * Short amino acid reads (30–50 aa)
  * Short nucleotide reads (90–150 nt)

The framework integrates multiple deep neural network architectures tailored for different input types and sequence lengths.

---

## 2. System Requirements

### Hardware (Mandatory)

* NVIDIA GPU
* CUDA-compatible device (CUDA ≥ 11.x recommended)
* ≥ 8 GB GPU memory recommended for long-sequence prediction

ARG_CMCRR **must be executed on GPU**. CPU-only execution is not supported.

To verify GPU availability:

```bash
nvidia-smi
```

Within Python:

```python
import torch
print(torch.cuda.is_available())
```

The output must be:

```
True
```
---

## 3. Installation

ARG_CMCRR uses Conda for environment management to ensure reproducibility.

### Step 1: Clone Repository

```bash
git clone https://github.com/hred-ux/ARG-CMCRR
cd ARG_CMCRR
```

### Step 2: Create Environment

A fully specified computational environment is provided via environment.yml.

Create the environment:

```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate arg_cmcrr
```
---

## 4. Usage

### General Command Structure

```
python ARG_CMCRR.py \
    --input <input_fasta> \
    --type <aa|nt> \
    --length <l|s> \
    --outfile <output_name> \
    --risk <True|False>
```
---

## 5. Supported Modes

### 5.1 Full-Length Amino Acid Sequences (Classification + Risk)

```
python ARG_CMCRR.py \
    --input input.fasta \
    --type aa \
    --length l \
    --outfile results \
    --risk True
```

### 5.2 Full-Length Amino Acid Sequences (Classification Only)

```
python ARG_CMCRR.py \
    --input input.fasta \
    --type aa \
    --length l \
    --outfile results \
    --risk False
```

### 5.3 Full-Length Nucleotide Sequences

```
python ARG_CMCRR.py \
    --input input.fasta \
    --type nt \
    --length l \
    --outfile results
```

### 5.4 Short Amino Acid Reads (30–50 aa)

```
python ARG_CMCRR.py \
    --input input.fasta \
    --type aa \
    --length s \
    --outfile results
```

### 5.5 Short Nucleotide Reads (90–150 nt)

```
python ARG_CMCRR.py \
    --input input.fasta \
    --type nt \
    --length s \
    --outfile results
```
---

## 6. Parameter Description

| Parameter           | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `--input` / `-i`    | Input FASTA file                                                   |
| `--type` / `-t`     | Molecular type (`aa` or `nt`)                                      |
| `--length` / `-l`   | Sequence type: `l` (long) or `s` (short)                           |
| `--outfile` / `-on` | Output file name                                                   |
| `--risk` / `-r`     | Enable ARG risk ranking (only valid for long amino acid sequences) |

---

## 7. Input Format

Standard FASTA format is required:

```
>sequence_1
MKTLLVAV...

```
---

## 8. Output

The output file includes:

* Header
* ARG prediction (ARG / non-ARG)
* ARG class prediction
* ARG mechanism prediction
* Risk level (if enabled)
* Blast_21_risk (if enabled risk)
* Blast_22_risk (if enabled risk)
---

## 9. Internal Model Routing

Depending on input parameters, ARG_CMCRR automatically dispatches prediction tasks to different modules:

| Input Configuration    | Module Invoked            |
| ---------------------- | ------------------------- |
| aa + short             | `argcmcrr_ssaa`           |
| nt + short             | `argcmcrr_ssnt`           |
| aa + long + risk=True  | `argcmcrr_lsaa_with_risk` |
| aa + long + risk=False | `argcmcrr_lsaa`           |
| nt + long              | `argcmcrr_lsnt`           |

---

## 10. Notes and Limitations

* GPU is mandatory.
* Risk prediction is supported **only for full-length amino acid sequences**.
* Input sequences outside recommended length ranges may affect performance.
* Ensure model `.pt` files are correctly placed in the `ptFile/` directory.

---

## 11. Citation

ARG_CMCRR 

---

## 12. Contact

For questions, bug reports, or collaboration inquiries, please open an Issue in the repository.

---


This tool is already structured at publication level — it only needs polishing for submission.

