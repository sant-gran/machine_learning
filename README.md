# Hybrid Recommendation System: Netflix + Facebook GNN

This project combines **Collaborative Filtering (CF)** with a **Graph Neural Network (GNN)** to build a hybrid recommendation engine using user behavior from Netflix and social connections from Facebook.

---

## Project Structure

```bash
machine_learning/
├── data_processing/              # Clean & preprocess raw data
│   ├── facebook/
│   │   ├── eda.py
│   │   └── preprocess.py
│   └── netflix/
│       ├── eda.py
│       └── preprocess.py
│
├── raw_data/                     # Downloaded datasets and encoders
│   └── netflix/
│       ├── netflix_ratings.csv
│       ├── netflix_normalized.csv
│       ├── movie_encoder.pkl
│       ├── user_encoder.pkl
│       └── ...
│
├── hybrid_model/                 # Core model logic
│   ├── cf_encoder.py             # CF (matrix-based or learned embeddings)
│   ├── gnn_encoder.py            # GNN encoding from Facebook data
│   ├── fusion.py                 # Combine CF + GNN (e.g., concat, MLP)
│   ├── predictor.py              # Predict scores from fused embeddings
│   └── hybrid_model.py           # Wraps the full hybrid architecture
│
├── training/                     # Training scripts
│   ├── train_cf.py               # CF model
│   ├── train_gnn.py              # GNN model
│   ├── train_hybrid.py           # End-to-end hybrid model
│   └── dataset.py                # Dataset + DataLoader logic
│
├── evaluation/                   # Evaluation scripts
│   ├── metrics.py                # Precision, recall, RMSE, etc.
│   └── evaluate_model.py         # Compare models
│
├── notebooks/                    # Jupyter notebooks for exploration
│   └── exploratory.ipynb
│
├── saved_models/                 # Trained models
│   ├── cf_encoder.pt
│   ├── gnn_encoder.pt
│   └── hybrid_model.pt
│
├── utils/                        # Helper utilities
│   ├── encoders.py               # Load/save LabelEncoders
│   └── logger.py                 # Training logs
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

##  Goal
To match users to personalized content by combining collaborative behavior and social context, improving recommendation accuracy compared to single-source systems.

---

##  Main Components
- **CFEncoder**: Learns user/movie embeddings from Netflix ratings
- **GNNEncoder**: Embeds users from the Facebook graph structure
- **Fusion Layer**: Combines both embeddings (concat or attention)
- **Predictor**: Outputs scores or top-N ranked items

---

##  How to Run

### 1. Preprocess Data
```bash
python data_processing/netflix/preprocess.py
python data_processing/facebook/preprocess.py
```

### 2. Train Models
```bash
python training/train_cf.py
python training/train_gnn.py
python training/train_hybrid.py
```

### 3. Evaluate Results
```bash
python evaluation/evaluate_model.py
```

---

## Metrics Used
- Precision@k
- Recall@k
- RMSE (for rating predictions)
- ROC AUC (for binary relevance)

---

##  Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

---

##  Contributions
Feel free to open issues or submit pull requests to improve data loaders, fusion strategies, or model performance.

---

## Contact
Created and maintained by Santiago Granados

