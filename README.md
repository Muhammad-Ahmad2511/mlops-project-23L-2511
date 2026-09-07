# MLOps Project - House Price Prediction

**Student ID:** 23L-2511

A minimal, reproducible MLOps workflow for a house price prediction model,
demonstrating clean separation of code, data, and model artifacts using
Git and GitHub.

## Project Structure

```
mlops-project-23L-2511/
├── data/                       # Raw dataset (ignored by git)
│   └── house_prices.csv
├── src/
│   └── train_23L-2511.py       # Training script
├── model/                      # Trained model output (ignored by git)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Clone the repository and move into the project root:

```bash
git clone https://github.com/<your-username>/mlops-project-23L-2511.git
cd mlops-project-23L-2511
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place your house price dataset as `data/house_prices.csv`. The CSV should
contain numeric feature columns and a target column named `price`.

## Running the Training Script

From the project root:

```bash
python src/train_23L-2511.py
```

This will:
1. Load the dataset from `data/house_prices.csv`
2. Train a `RandomForestRegressor`
3. Print evaluation metrics (MAE, R2)
4. Save the trained model to `model/house_price_model_23L-2511.pkl`

## Notes

- Raw data and trained model files are excluded from version control via
  `.gitignore` to keep the repository lightweight.
- Only source code (`.py`) and configuration files (`.txt`, `.md`,
  `.gitignore`) are tracked in this repository.
