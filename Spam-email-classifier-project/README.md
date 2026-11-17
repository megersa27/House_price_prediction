# Spam Email Classifier

This project builds a simple spam email detector using a Naive Bayes classifier and TF-IDF features from scikit-learn.

## What it does

- Loads a small labeled dataset of spam and ham messages
- Converts text into TF-IDF features
- Trains a Multinomial Naive Bayes model
- Evaluates accuracy on a test split
- Predicts labels for new sample messages

## Run it locally

```bash
cd Spam-email-classifier-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 spam_test.py
```

## Files

- `spam_test.py` — training, evaluation, and prediction logic
- `requirements.txt` — project dependencies
