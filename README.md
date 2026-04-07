# Video Game Sales Analysis & Prediction

A clean, modular, and well-structured machine learning project for analyzing and predicting video game sales.

---

## Project Workflow Tree

```
Video Game Sales Analysis & Prediction
│
├──  Data Collection
│   ├── Source: Video Game Sales Dataset (vgsales.csv)
│   └── Attributes: Rank, Name, Platform, Year, Genre, Publisher,
│                   NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales
│
├──  Data Preprocessing
│   ├── Missing value removal (dropna)
│   ├── Year filtering (≤ 2015)
│   ├── Column removal (Name)
│   └── Data type validation
│
├──  Exploratory Data Analysis (EDA)
│   ├── Sales distribution analysis
│   │   └── Histograms with gamma fit
│   ├── Platform-wise comparison
│   │   └── Bar charts for global sales by platform
│   ├── Genre-wise trends
│   │   └── Count plots and sales aggregations
│   ├── Regional sales comparison
│   │   └── Heatmaps and grouped bar charts
│   ├── Publisher analysis
│   │   └── Top publishers by count and sales
│   └── Correlation analysis
│       └── Correlation heatmap for numerical features
│
├──  Feature Encoding
│   ├── Method: Label Encoding (sklearn.LabelEncoder)
│   ├── Platform → Numeric (0 to n-1)
│   ├── Genre → Numeric (0 to n-1)
│   └── Publisher → Numeric (0 to n-1)
│
├──  Model Training
│   ├── Random Forest Regressor
│   │   ├── n_estimators: 300
│   │   └── max_depth: 20
│   ├── XGBoost Regressor
│   │   ├── n_estimators: 300
│   │   ├── learning_rate: 0.05
│   │   ├── max_depth: 8
│   │   └── subsample: 0.8
│   └── Hybrid Ensemble Model
│       └── Average of RF + XGBoost predictions
│
├──  Model Evaluation
│   ├── R² Score (Coefficient of Determination)
│   ├── MAE (Mean Absolute Error)
│   └── RMSE (Root Mean Squared Error)
│
└──  Prediction & Output
    ├── Global Sales Prediction
    ├── Model Comparison Output
    └── Best Model Recommendation
```

---

##  Project Structure

```
video_game_sales_analysis/
│
├── config.py              # Configuration constants & hyperparameters
├── data_collection.py     # Data loading module
├── data_preprocessing.py  # Data cleaning & filtering
├── eda.py                 # Exploratory Data Analysis
├── feature_encoding.py    # Label encoding utilities
├── model_training.py      # RF, XGBoost, Hybrid training
├── model_evaluation.py    # Evaluation metrics
├── prediction.py          # Prediction output module
├── main.py                # Main orchestration script
└── README.md              # This file
```

---

##  Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost missingno
```

### 2. Place Your Dataset

Update `DATA_PATH` in `config.py` to point to your `vgsales.csv` file.

### 3. Run the Pipeline

```bash
python main.py
```

### 4. Run with EDA Visualizations

```python
from main import main
results, evaluations = main(run_eda=True)
```

---

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `config.py` | Central configuration file with all constants |
| `data_collection.py` | Load dataset and display basic info |
| `data_preprocessing.py` | Clean, filter, and prepare data |
| `eda.py` | Generate visualizations and insights |
| `feature_encoding.py` | Convert categorical features to numeric |
| `model_training.py` | Train ML models (RF, XGBoost, Hybrid) |
| `model_evaluation.py` | Calculate R², MAE, RMSE metrics |
| `prediction.py` | Generate and display predictions |
| `main.py` | Orchestrate the complete pipeline |

---

## Sample Output

```
╔══════════════════════════════════════════════════════════╗
║          VIDEO GAME SALES ANALYSIS & PREDICTION          ║
╚══════════════════════════════════════════════════════════╝

▶ STEP 1: DATA COLLECTION
▶ STEP 2: DATA PREPROCESSING
▶ STEP 3: EDA (optional)
▶ STEP 4: FEATURE ENCODING
▶ STEP 5: MODEL TRAINING
▶ STEP 6: MODEL EVALUATION
▶ STEP 7: PREDICTION & OUTPUT

═══════════════════════════════════════════════════════════
MODEL COMPARISON SUMMARY
═══════════════════════════════════════════════════════════

Model                         R² Score          MAE         RMSE
───────────────────────────────────────────────────────────────
Random Forest                   0.XXXX       0.XXXX       0.XXXX
XGBoost                         0.XXXX       0.XXXX       0.XXXX
Hybrid (RF + XGB)               0.XXXX       0.XXXX       0.XXXX

 Best Model: [Model Name] (R² = 0.XXXX)

 PIPELINE COMPLETE!
```

---

## Notes

- All original algorithms and logic are preserved
- Only the code organization has been improved
- EDA visualizations are optional (set `run_eda=True` to enable)
- Hyperparameters can be modified in `config.py`

---

## License

This project is for educational and academic purposes.
