import pyJASPAR
import inMOTIFin
import numpy as np
from .example_sequence_features import one_hot_encode

"""WARNING: COPILOT GENERATED THIS CODE"""





def get_jaspar_motif(motif_id):
    """
    Fetch a JASPAR motif by ID and return its PWM.
    
    Args:
        motif_id: JASPAR motif ID (e.g., 'MA0001.1')
        
    Returns:
        numpy array of shape (4, motif_length) for ACGT
    """
    jdb = pyJASPAR.JASPAR()
    motif = jdb.fetch_motif_by_id(motif_id)
    pwm = motif.pwm
    # Convert to numpy array in order A, C, G, T
    pwm_array = np.array([pwm['A'], pwm['C'], pwm['G'], pwm['T']])
    return pwm_array


def compute_motif_convolution(sequence, pwm):
    """
    Compute convolution scores of a PWM with a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        pwm: Position Weight Matrix as numpy array (4, motif_length)
        
    Returns:
        numpy array of convolution scores along the sequence
    """
    seq_encoded = one_hot_encode(sequence)  # shape (seq_len, 4)
    motif_length = pwm.shape[1]
    seq_length = len(sequence)
    
    scores = []
    for i in range(seq_length - motif_length + 1):
        window = seq_encoded[i:i+motif_length]  # shape (motif_length, 4)
        score = np.sum(window * pwm.T)  # dot product
        scores.append(score)
    
    return np.array(scores)


def compute_jaspar_features(sequences, motif_ids):
    """
    Compute JASPAR motif convolution features for a list of sequences.
    
    Args:
        sequences: List of DNA sequence strings
        motif_ids: List of JASPAR motif IDs
        
    Returns:
        Dictionary with motif scores for each sequence
    """
    features = {}
    
    for motif_id in motif_ids:
        pwm = get_jaspar_motif(motif_id)
        motif_name = f"motif_{motif_id.replace('.', '_')}"
        features[motif_name] = []
        
        for seq in sequences:
            scores = compute_motif_convolution(seq, pwm)
            # Aggregate scores: max, mean, sum
            features[motif_name].append({
                'max_score': np.max(scores),
                'mean_score': np.mean(scores),
                'sum_score': np.sum(scores)
            })
    
    return features


# Example usage
if __name__ == "__main__":
    # Test with sample data
    test_sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    test_motifs = ["MA0001.1"]  # Example motif ID
    
    features = compute_jaspar_features(test_sequences, test_motifs)
    print(features)
