"""Data loading and preprocessing utilities."""

import os
import pandas as pd
import numpy as np
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_fasta(fasta_path):
    """
    Load sequences from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Dictionary with sequence IDs and sequences
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def load_dataset(csv_path):
    """
    Load dataset from CSV with sequences and expression levels.
    
    Args:
        csv_path: Path to CSV file with columns: 'sequence' and 'expression'
        
    Returns:
        pandas DataFrame
    """
    return pd.read_csv(csv_path)


def validate_sequences(sequences):
    """
    Validate DNA sequences (only ACGT characters).
    
    Args:
        sequences: List or array of sequences
        
    Returns:
        Cleaned sequences, list of invalid indices
    """
    valid_sequences = []
    invalid_indices = []
    
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        if all(n in 'ACGT' for n in seq):
            valid_sequences.append(seq)
        else:
            invalid_indices.append(i)
    
    return valid_sequences, invalid_indices


def save_processed_data(X, y, output_prefix):
    """
    Save processed features and targets.
    
    Args:
        X: Feature matrix
        y: Target values
        output_prefix: Prefix for output files
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'{output_prefix}_X.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'{output_prefix}_y.npy'), y)


def load_processed_data(output_prefix):
    """
    Load processed features and targets.
    
    Args:
        output_prefix: Prefix for input files
        
    Returns:
        Tuple of (X, y)
    """
    X = np.load(os.path.join(PROCESSED_DATA_DIR, f'{output_prefix}_X.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, f'{output_prefix}_y.npy'))
    return X, y
