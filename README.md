# house-price-prediction

[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) [![protected: by-gitleaks-blue](https://img.shields.io/badge/protected%20by-gitleaks-blue)](https://github.com/gitleaks/gitleaks-action) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Uses the Cookiecutter Data Science project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

This project demonstrates how to properly clean data *(handle missing, handle encoding, handle numerical data)*, train a model *(using a regression model)*, and predict and serve it through API.

---

## 🛠 Technologies Used

- Python 3.10+
- Pandas, NumPy, Scikit-Learn, JobLib, LightGBM, MatPlotLib, XGBoost
- FastAPI, Uvicorn, Pydantic, Streamlit
- GitHub

---

## 📁 Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         house_price_prediction and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── tests              <- Tests to run against modules, models and app.
│   └── test_app.py    <- Tests on streamlit execution.
│
├── house_price_prediction   <- Source code for use in this project.
│   │
│   ├── __init__.py             <- Makes house_price_prediction a Python module
│   │
│   ├── config.py               <- Store useful variables and configuration
│   │
│   ├── dataset.py              <- Scripts to download or generate data
│   │
│   ├── modeling                
│   │   ├── __init__.py 
│   │   ├── predict.py          <- Code to run model inference with trained models          
│   │   ├── train.py            <- Code to train models          
│   │   └── schemas.py          <- Code of the @dataclass and pydantic models
│   │
│   └── plots.py                <- Code to create visualizations
│
├── api.py             <- Code to run the FastAPI with a predict route
│
└── api.py             <- Code to run the Streamlit app
```

---

## 🚀 How to Run

### Prerequisites

1. Put the kaggle.json file in the .kaggle in the root folder, it must contain 2 fields : `username` and `key` *(which is the API key)*.

### Run it

1. Clone the repo  
2. Install dependencies  
3. Build the processed dataset  
```bash
make requirements
make process
make train
make plots
```
4. If you want to predict, then use  
```bash
make predict
```
5. If you want to use the API instead, use  
```bash
make api
```  
And hit [the local URL](http://127.0.0.1:8000/docs) in your browser.
6. If you want to run the streamlit app locally, use *(not functional yet)*  
```bash
make app
```

### Test it

1. Simply run the pytest using  
```bash
make test
```