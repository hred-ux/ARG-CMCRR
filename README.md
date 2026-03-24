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
git clone https://github.com/hred-ux/ARG-CMCRR.git
cd ARG-CMCRR
```

### Step 2: Create Environment

A fully specified computational environment is provided via environment.yml.

Create the environment:

```bash
conda env create -n arg_cmcrr -f environment.yml
```

Activate the environment:
```bash
conda activate arg_cmcrr
```
---
## 4. Download Required External Resources

ARG_CMCRR requires two external resources:

* ESM-2 protein language model

* NCBI BLAST+

These resources must be downloaded manually before running the program.

### 4.1 Download ESM-2 Model

ARG_CMCRR uses the ESM-2 (650M) protein language model.

Download from HuggingFace:
```bash
https://huggingface.co/facebook/esm2_t33_650M_UR50D
```
After downloading, place the model files in the following directory:
```bash
ARG_CMCRR/models/esm2_t33_650M/
```
### 4.2 Install NCBI BLAST+

Download NCBI BLAST+ from:
```bash
https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
```
After downloading and extracting, place the BLAST binaries in:
```bash
ARG_CMCRR/blast/bin/
```
## 5. Usage

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

## 6. Supported Modes

### 6.1 Full-Length Amino Acid Sequences (Classification + Risk)

```
python ARG_CMCRR.py --input input.fasta  --type aa  --length l   --outfile results   --risk True
```

### 6.2 Full-Length Amino Acid Sequences (Classification Only)

```
python ARG_CMCRR.py --input input.fasta  --type aa  --length l   --outfile results   --risk False
```

### 6.3 Full-Length Nucleotide Sequences

```
python ARG_CMCRR.py --input input.fasta  --type nt  --length l   --outfile results  
```

### 6.4 Short Amino Acid Reads

```
python ARG_CMCRR.py --input input.fasta  --type aa  --length s   --outfile results 
```

### 6.5 Short Nucleotide Reads 

```
python ARG_CMCRR.py --input input.fasta  --type nt  --length s   --outfile results 
```
---

## 7. Parameter Description

| Parameter           | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `--input` / `-i`    | Input FASTA file                                                   |
| `--type` / `-t`     | Molecular type (`aa` or `nt`)                                      |
| `--length` / `-l`   | Sequence type: `l` (long) or `s` (short)                           |
| `--outfile` / `-on` | Output file name                                                   |
| `--risk` / `-r`     | Enable ARG risk ranking (only valid for long amino acid sequences) |

---

## 8. Input Format

Standard FASTA format is required:

```
>sequence_1
MKTLLVAV...

```
---

## 9. Output

The output file includes:

* Header
* ARG prediction (ARG / non-ARG)
* ARG class prediction
* ARG mechanism prediction
* Risk level (if enabled)
* Blast_21_risk (if enabled risk)
* Blast_22_risk (if enabled risk)
---

## 10. Internal Model Routing

Depending on input parameters, ARG_CMCRR automatically dispatches prediction tasks to different modules:

| Input Configuration    | Module Invoked            |
| ---------------------- | ------------------------- |
| aa + short             | `argcmcrr_ssaa`           |
| nt + short             | `argcmcrr_ssnt`           |
| aa + long + risk=True  | `argcmcrr_lsaa_with_risk` |
| aa + long + risk=False | `argcmcrr_lsaa`           |
| nt + long              | `argcmcrr_lsnt`           |

---

## 11. Notes and Limitations

* GPU is mandatory.
* Risk prediction is supported **only for full-length amino acid sequences**.
* Input sequences outside recommended length ranges may affect performance.
* Ensure model `.pt` files are correctly placed in the `ptFile/` directory.

---

## 12. Citation

ARG_CMCRR 

---

## 13. Contact

For questions, bug reports, or collaboration inquiries, please open an Issue in the repository.

---


This tool is already structured at publication level — it only needs polishing for submission.

