# NATOPS-TSC: Time Series Classification with LSTM

This repository implements an LSTM-based time series classification (TSC) model for the NATOPS dataset.

## 🚀 Project Overview
- **Dataset:** NATOPS (24D hand movement coordinates, 6 gesture classes)
- **Model:** LSTM-based classifier
- **Optimization:** Grid search for hyperparameters
- **Frameworks:** PyTorch, tslearn, scikit-learn

# 0. Open the implementation.ipynb file and keep on running the cells one after another.

# 1️.  Install dependencies
pip install -r requirements.txt

# 2. Preprocess dataset
python training/preprocess.py

# 3. Train the LSTM Model
python training/train.py

# 4. Hyperparameter tuning (Grid search)
python training/grid_search.py

# 5. Run inference
python inference/inference_final.py






