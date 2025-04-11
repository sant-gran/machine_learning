# Hybrid Recommendation System: Netflix + Facebook GNN

This project combines **Collaborative Filtering (CF)** with a **Graph Neural Network (GNN)** to build a hybrid recommendation engine using user behavior from Netflix and social connections from Facebook.

---

## Project Structure

```bash
neurograph/
├── data_processing/
│   ├── facebook/
│   │   ├── eda.py
│   │   └── preprocess.py
│   └── netflix/
│       ├── eda.py
│       └── preprocess.py
├── raw_data/
│   └── netflix/
│       ├── netflix_ratings.csv
│       ├── netflix_normalized.csv
│       ├── movie_encoder.pkl
│       ├── user_encoder.pkl
├── hybrid_model/
│   ├── cf_encoder.py
│   ├── gnn_encoder.py
│   ├── fusion.py
│   ├── predictor.py
│   └── hybrid_model.py
├── training/
│   ├── train_cf.py
│   ├── train_gnn.py
│   ├── train_hybrid.py
│   └── dataset.py
├── evaluation/
│   ├── metrics.py
│   └── evaluate_model.py
├── notebooks/
│   └── exploratory.ipynb
├── saved_models/
│   ├── cf_encoder.pt
│   ├── gnn_encoder.pt
│   └── hybrid_model.pt
├── utils/
│   ├── encoders.py
│   └── logger.py
├── requirements.txt
└── README.md

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
Created and maintained by the Mancii ML Team.

