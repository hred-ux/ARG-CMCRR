import sys
from esm import FastaBatchedDataset
from Bio.Seq import Seq
import os
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def process_ssaa(fasta_file):
    max_seq_len = 40
    dataset = FastaBatchedDataset.from_file(fasta_file)
    sequences = dataset.sequence_strs
    names = []

    for label in dataset.sequence_labels:
        names.append(label)
    query_x, actual_len = seq_one_hot(sequences, max_seq_len)

    return query_x, actual_len, names


def process_lsaa(fasta_file):
    max_seq_len = 400
    dataset = FastaBatchedDataset.from_file(fasta_file)
    sequences = dataset.sequence_strs
    names = []
    sequencesList = []

    for label,seq in zip(dataset.sequence_labels,sequences):
        len_seq = len(seq)
        if len_seq < 50:
            continue
        else:
            names.append(label)
            sequencesList.append(seq)
    query_x, actual_len = seq_one_hot(sequencesList, max_seq_len)

    return query_x, actual_len, names, sequencesList

def process_lsnt(fasta_file):
    max_seq_len = 400
    inf = open(fasta_file, 'r')
    dict = fasta2dict(inf)
    inf.close()

    key_list = list(dict.keys())
    seq_list = list(dict.values())
    sequences = []
    names = []

    for header, seq in zip(key_list, seq_list):
        six_frames = translate_six_frames(seq)
        longest_seq = ''
        for frame in six_frames:
            if '*' not in frame:
                longest_seq = frame
                break
            else:
                for sub_seq in frame.split('*'):
                    if len(sub_seq) > len(longest_seq):
                        longest_seq = sub_seq
        if len(longest_seq) < 50:
            continue
        else:
            sequences.append(longest_seq)
            names.append(header)
    query_x, actual_len = seq_one_hot(sequences, max_seq_len)

    return query_x, actual_len, names


def process_ssnt(fasta_file):
    max_seq_len = 40
    inf = open(fasta_file, 'r')
    dict = fasta2dict(inf)
    inf.close()

    key_list = list(dict.keys())
    seq_list = list(dict.values())
    sequences = []
    names = []

    for header, seq in zip(key_list, seq_list):
        six_frames = translate_six_frames(seq)
        longest_seq = ''
        for frame in six_frames:
            if '*' not in frame:
                longest_seq = frame
                break
            else:
                for sub_seq in frame.split('*'):
                    if len(sub_seq) > len(longest_seq):
                        longest_seq = sub_seq
        if len(longest_seq) < 30:
            continue
        else:
            sequences.append(longest_seq)
            names.append(header)
    query_x, actual_len = seq_one_hot(sequences, max_seq_len)

    return query_x, actual_len, names

def fasta2dict(inf):
    dict = {}
    for line in inf:
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]
            dict[name] = ''
        else:
            dict[name] += line
    return dict

def translate_six_frames(dna_sequence):
      dna_seq = Seq(dna_sequence)
      frames = []

      for frame in range(3):
            seq = dna_seq[frame:]
            trimmed_seq = seq + "N" * (3 - (len(seq) % 3))
            protein = trimmed_seq.translate(to_stop=False).rstrip('*')
            frames.append(f"{protein}")

      reverse_dna_seq = dna_seq.reverse_complement()

      for frame in range(3):
            seq = reverse_dna_seq[frame:]
            trimmed_seq = seq + "N" * (3 - (len(seq) % 3))
            protein = trimmed_seq.translate(to_stop=False).rstrip('*')
            frames.append(f"{protein}")

      return frames

def seq_one_hot(sequence, max_seq_len):
    seq_len = []
    sequence_indices = []
    length = len(sequence)
    actual_len = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(length):
        seq_len.append(len(sequence[i]))
    aa_to_index = {'<pad>':0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
                   'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16,
                   'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21}
    for seq in sequence:    
        indices = torch.tensor([aa_to_index[aa] if aa in aa_to_index else 0 for aa in seq[:max_seq_len]], dtype=torch.long).to(device)
        if len(indices) < max_seq_len:
            padded_indices = torch.cat([indices, torch.zeros(max_seq_len - len(indices), dtype=torch.long).to(device)])
            actual_len.append(len(indices))
        else:
            padded_indices = indices
            actual_len.append(max_seq_len)
        sequence_indices.append(padded_indices)

    padded_sequence = torch.stack(sequence_indices)
    
    return padded_sequence, actual_len


