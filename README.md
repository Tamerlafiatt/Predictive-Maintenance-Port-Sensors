# Port Equipment Maintenance Time Prediction

Predicting repair durations for port equipment (cranes, stackers, spreaders) using **Random Forest** and **Bagging Regressor**, with a deployed **Streamlit** web app.

## Problem

Given an equipment type and fault category, predict the number of hours needed to complete the repair intervention — enabling better maintenance scheduling and resource allocation.

## Dataset

`donnees_tiers.xlsx` — 8,500 maintenance interventions, synthetic data based on real port operations.

| Column | Description |
|---|---|
| Libellé Equipement | Equipment type (12 unique: LHM400, TEREX, BROMMA…) |
| Titre D.I | Fault category (6 types: roue, moteur, fuite…) |
| H. réalisées | **Target**: actual repair hours (1–25h, mean=8.2h) |
| Date D.I | Intervention request date |
| Date debut/fin travail | Work start/end timestamps |

## Models

| Model | MAE | Notes |
|---|---|---|
| RandomForest | ~3.1h | n_estimators tuned via OOB sweep (10→200) |
| Bagging(RF) | ~3.2h | Ensemble of RF estimators |

**Key design decisions:**
- **OneHotEncoding** over LabelEncoding: avoids imposing ordinal relationships between equipment types
- **OOB error curve** used to select n_estimators without a separate validation set
- **Date features** (month, day of week) extracted but not used in final model — equipment+fault type alone gives best MAE

## Project Structure

```
├── train.py          # Data loading, encoding, RF/Bagging training, hyperparameter search
├── visualize.py      # EDA plots: distributions, fault breakdown, equipment analysis
├── app.py            # Streamlit prediction UI
├── requirements.txt
├── donnees_tiers.xlsx
├── notebooks/
│   └── results.ipynb
└── results/          # Saved plots and model files
```

## Quick Start

```bash
pip install -r requirements.txt

# Train and save model
python train.py

# Generate EDA plots
python visualize.py

# Launch web app
streamlit run app.py
```

## License

MIT
