# BalanceIT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#)
[![Django 4.x](https://img.shields.io/badge/Django-4.x-092E20?logo=django&logoColor=white)](#)
[![Node.js 18+](https://img.shields.io/badge/Node.js-18%2B-339933?logo=node.js&logoColor=white)](#)
[![Tailwind CSS 3.x](https://img.shields.io/badge/Tailwind_CSS-3.x-06B6D4?logo=tailwind-css&logoColor=white)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#)

An interactive Django application for analyzing imbalanced text datasets and training classifiers with resampling techniques.

## Features
- Analyze class imbalance with multiple metrics (IR, KL, TV, Hellinger, etc.)
- Wizard-based workflow to pick dataset, target, sampler, and model
- Supports scikit-learn, imbalanced-learn, imbalanced-ensemble, XGBoost, LightGBM
- TF-IDF vectorization, train/test split, model training, and evaluation
- Stores results and datasets per user; login/register flows
- Tailwind CSS UI with ApexCharts visualizations

## UI

Below are snapshots of the BalanceIT interface:

<div align="center">
  <img src="screenshots/customize.png" alt="Customize screen" width="420" />
  <img src="screenshots/customize2.png" alt="Customize step 2" width="420" />
</div>

<div align="center">
  <img src="screenshots/customize3.png" alt="Customize step 3" width="420" />
  <img src="screenshots/customize4.png" alt="Customize step 4" width="420" />
</div>

<div align="center">
  <img src="screenshots/customize5.png" alt="Customize step 5" width="420" />
  <img src="screenshots/customize6.png" alt="Customize step 6" width="420" />
</div>

<div align="center">
  <img src="screenshots/customize7.png" alt="Customize step 7" width="420" />
  <img src="screenshots/tut14.png" alt="Tutorial view" width="420" />
</div>

## Tech Stack
- Backend: Django 4, django-crispy-forms, django-formtools
- ML: scikit-learn, imbalanced-learn, imbalanced-ensemble, xgboost, lightgbm, nltk, pandas, numpy
- Frontend: Tailwind CSS, PostCSS, Autoprefixer, ApexCharts

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1) Python setup
```bash
python -m venv .venv
# PowerShell
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Apply migrations and run the server:
```bash
python manage.py migrate
python manage.py createsuperuser  # optional
python manage.py runserver
```

### 2) Frontend (optional, for CSS builds)
```bash
npm install
# Build Tailwind CSS once
npx tailwindcss -i ./src/styles.css -o ./static/css/styles.css --minify
# Or watch during development
npx tailwindcss -i ./src/styles.css -o ./static/css/styles.css --watch
```

If Tailwind build errors about missing plugins (e.g., Flowbite), install them:
```bash
npm i flowbite
```

### 3) Environment variables
Copy `.env.example` to `.env` and adjust as needed:
```bash
# PowerShell
copy .env.example .env
```
The app reads:
- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG` ("True"/"False")
- `DJANGO_ALLOWED_HOSTS` (comma-separated)

## Datasets
- Upload CSVs via the UI (recommended)
- The `media/` folder is git-ignored to avoid committing large datasets

## License
MIT. See `LICENSE`.
