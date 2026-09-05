## 6.6 Demo of Applying the Model to Metagenomic Analysis

If you want to begin your analysis with raw metagenomic sequencing reads, you can follow the workflow below to run predictions with ARG-CMCRR.

ARG-CMCRR supports the prediction of short nucleotide sequences. Raw metagenomic sequencing data are commonly provided in FASTQ format, whereas ARG-CMCRR accepts nucleotide sequences in FASTA format. Therefore, FASTQ reads can first be converted to FASTA format using commonly used bioinformatics tools such as Seqtk, followed by prediction using the short-nucleotide mode of ARG-CMCRR.

### 6.6.1 Single-end sequencing

For single-end metagenomic reads:

```bash
# Convert file format from FASTQ to FASTA
seqtk seq -a sample.fastq.gz > sample.fasta

# Remove duplicated sequences by sequence
seqkit rmdup -s -i sample.fasta -o unique.sample.fasta
```

The processed reads can then be analyzed using ARG-CMCRR:

```bash
python ARG-CMCRR.py \
    --input unique.sample.fasta \
    --type nt \
    --length s \
    --outfile sample_results
```

### 6.6.2 Paired-end sequencing

For paired-end sequencing data, R1 and R2 reads can be converted separately:

```bash
seqtk seq -a sample_R1.fastq.gz > sample_R1.fasta
seqtk seq -a sample_R2.fastq.gz > sample_R2.fasta
```

The two read files can then be merged by:

```bash
seqtk mergefa sample_R1.fasta sample_R2.fasta > sample_merged_R1R2.fasta
```

Remove duplicated sequences:

```bash
seqkit rmdup -s -i sample_merged_R1R2.fasta -o unique.sample_merged_R1R2.fasta
```

The processed reads can then be analyzed using ARG-CMCRR:

```bash
python ARG-CMCRR.py \
    --input unique.sample_merged_R1R2.fasta \
    --type nt \
    --length s \
    --outfile sample_results
```

This workflow enables ARG-CMCRR to be applied directly downstream of raw metagenomic sequencing data without modifying the prediction model.
