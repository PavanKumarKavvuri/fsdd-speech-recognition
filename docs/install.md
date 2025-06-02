# Installation Instructions

This guide provides step-by-step instructions to set up the environment for running the Spoken Digit Classification project.


## Project Structure

```bash
├── notebooks/ # Jupyter notebooks for step-by-step exploration
│ ├── Task-A.ipynb # Features Extraction, and Dataset preparation
│ ├── Task-B.ipynb # LSTM model definitions
│ └── Task-C.ipynb # Helpers
├── docs/ 
│ └── install.md # Installation instructions
├── src/ # Modular Python source files
│ ├── data_preprocessor.py # Features Extraction, and Dataset preparation
│ ├── model.py # LSTM model definitions
│ ├── train.py # Training logic
│ ├── evaluate.py # Evaluation logic
│ ├── quantize.py # Quantization and memory-constrained versions
│ └── utils.py # Helpers
├── config/ # YAML-based experiment configs
│ └── model_config.yaml # Model and Experiments hyperparameters
├── outputs/ # Saved models, logs, figures
├── README.md # Assignment documentation
├── requirements.txt # Dependencies
└── LICENCE 
```

## 1. Create and Activate Virtual Environment

It's recommended to use a virtual environment to isolate project dependencies:

```bash
# Create virtual environment (Python 3.8+ recommended)
python -m venv fsdd_assignment

# Activate the environment
# On Linux:
source fsdd_assignment/bin/activate

# On Windows:
fsdd_assignment\Scripts\activate
```

## 2. Clone the repository

```bash
git clone git@github.com:PavanKumarKavvuri/fsdd-speech-recognition.git
cd fsdd-speech-recognition
```

## 3. Install dependencies
Once the virtual environment is activated, install required packages using pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Next Steps

After installation, you can start by:
- Running audio data inspection: `jupyter notebook notebooks/audio_data_analysis.ipynb`
- Task A: `jupyter notebook notebooks/audio_data_analysis.ipynb`
- Task B(Memory Constraint): `jupyter notebook notebooks/audio_data_analysis.ipynb`
- Task B(Int8 Quantisation): `jupyter notebook notebooks/audio_data_analysis.ipynb`
- Task C(Po2 Quantisation): `jupyter notebook notebooks/audio_data_analysis.ipynb`



