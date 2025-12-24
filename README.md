# Promoter Sequence Gene Expression Prediction

A machine learning project to predict gene expression levels from promoter DNA sequences using various sequence features and ML algorithms.

## Project Overview

This project builds and evaluates machine learning models to predict gene expression levels based on promoter sequence data. The workflow includes:
- Data exploration and sequence analysis
- Feature engineering from DNA sequences
- Model training and evaluation
- Prediction on new sequences

## Project Structure

```
.
├── data/
│   ├── raw/                 # Raw sequence data (FASTA or CSV)
│   └── processed/          # Processed features (NumPy arrays)
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── features/           # Feature extraction modules
│   │   └── sequence_features.py
│   ├── models/             # Model training utilities
│   │   └── train.py
│   └── utils/              # Data loading and processing
│       └── data_loader.py
├── models/                 # Trained model files
├── tests/                  # Unit tests
├── config.py               # Configuration file
├── requirements.txt        # Python dependencies
└── setup.py               # Package setup file
```

## Key Features

### Feature Engineering
- **Nucleotide Composition**: A, C, G, T frequencies and GC content
- **Dinucleotide Features**: 16 dinucleotide frequencies (AA, AC, AG, etc.)
- **K-mer Analysis**: K-mer frequency distributions
- **One-hot Encoding**: Direct sequence encoding

### Models Implemented
- **Ridge Regression**: Linear model with L2 regularization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree boosting

### Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Cross-validation scores

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd intro_to_igem_challenge
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Explore Data**: Run `01_exploratory_data_analysis.ipynb`
   - Load and visualize promoter sequences
   - Analyze sequence composition
   - Examine expression level distributions
   - Identify feature correlations

2. **Engineer Features**: Run `02_feature_engineering.ipynb`
   - Extract various sequence features
   - Combine features for modeling
   - Save processed data

3. **Train Models**: Run `03_model_training.ipynb`
   - Train baseline models
   - Evaluate performance
   - Compare models
   - Save best model

## Data Format

### Input Data (CSV)
```csv
sequence,expression
ACGTACGTACGT...,7.5
GCTAGCTAGCTA...,5.2
```

### Sequence Requirements
- Must contain only valid DNA nucleotides (A, C, G, T)
- Recommended length: 50-500 bp
- Handle multiple sequences efficiently

## Configuration

Edit `config.py` to customize:
- Sequence length
- Train/test split ratios
- Model hyperparameters
- K-mer size for feature extraction
- Learning rates and batch sizes

## Usage

### As a Python Package

```python
from src.features.sequence_features import nucleotide_composition
from src.utils.data_loader import load_dataset
from src.models.train import train_model

# Load data
df = load_dataset('data/raw/promoters.csv')

# Extract features
features = [nucleotide_composition(seq) for seq in df['sequence']]

# Train model
model, scaler, train_r2, test_r2 = train_model(X, y, ridge_model)
```

### Making Predictions

```python
from src.models.train import load_model
from src.features.sequence_features import nucleotide_composition

# Load trained model
model, scaler = load_model('models/best_model.joblib', 'models/best_scaler.joblib')

# Predict on new sequence
new_seq = "ACGTACGTACGT..."
features = nucleotide_composition(new_seq)
expression = model.predict(scaler.transform([features]))
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras (optional, for deep learning)
- Matplotlib & Seaborn (for visualization)
- Jupyter

## Next Steps

### Enhancement Ideas
1. **Deep Learning**: Implement CNN or LSTM models for sequence analysis
2. **Transfer Learning**: Use pre-trained DNA language models
3. **Feature Selection**: Identify most important features via SHAP values
4. **Hyperparameter Optimization**: Use Bayesian optimization
5. **Ensemble Methods**: Stack multiple models for better predictions
6. **Cross-species Analysis**: Compare predictions across organisms

### Data Sources
- [NCBI Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/)
- [iGEM Parts Registry](http://parts.igem.org/)
- Custom experimental data

## Contributing

To contribute to this project:
1. Create a new branch for your feature
2. Make your changes
3. Add tests in the `tests/` directory
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## References

- Koo, T., & Eddy, S. R. (2019). Evolving regulatory sequences in prokaryotes. *Current Opinion in Systems Biology*, 15, 1-7.
- Osmanbeyoglu, H. U., et al. (2019). Pathways of Concerted Tumor Suppressor Inactivation in the Context of VHL Loss in Renal Cancer. *Cell Reports*, 8(5), 1346-1361.

## Contact

For questions or feedback, please open an issue on the repository or contact the maintainers.