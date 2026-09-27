# OPay App Review Sentiment Analysis

Sentiment analysis of OPay Google Play Store reviews. Reviews are scraped, cleaned, explored, and used to train and compare three machine learning models that classify reviews as **positive** or **negative**.

## Overview

- **Data source:** 5,000 reviews for the OPay app (`team.opay.pay`) scraped from the Google Play Store using `google-play-scraper`.
- **Labeling:** Star ratings are converted into binary sentiment labels — scores of 3–5 are labeled **positive (1)**, scores of 1–2 are labeled **negative (0)**.
- **Goal:** Build a text classifier that predicts whether a review is positive or negative from its written content.

## Project Workflow

1. **Data Collection** — Scrape reviews and app metadata via `google_play_scraper` and save to `opay_reviews_raw.csv`.
2. **Data Cleaning** — Check for duplicates and missing values, impute/drop as needed, drop unused columns (`reviewId`, `userName`, `userImage`, `replyContent`, `appVersion`), and convert timestamp columns to datetime.
3. **Exploratory Data Analysis (EDA)**
   - Word clouds of the most common words overall, and separately for positive vs. negative reviews.
   - Review length distribution.
   - Score distribution (pie chart).
   - Manual inspection of sample reviews per score to justify the labeling threshold.
   - Analysis of `thumbsUpCount` to surface the most "liked" complaints.
   - Sentiment breakdown by app version (`reviewCreatedVersion`).
4. **Text Preprocessing** — Lowercasing, URL removal, tokenization, stopword removal (keeping negators like *not*, *no*, *never*), and lemmatization via NLTK.
5. **Feature Extraction** — TF-IDF vectorization (`max_features=5000`, unigrams + bigrams).
6. **Modeling** — Train/test split (80/20, stratified) and train three classifiers:
   - Logistic Regression (`class_weight="balanced"`)
   - Multinomial Naive Bayes
   - Linear Support Vector Classifier (`class_weight="balanced"`)
7. **Evaluation** — Classification reports and confusion matrices for each model, plus qualitative testing on hand-written sample reviews.
8. **Model Persistence** — Trained models and the TF-IDF vectorizer are saved as `.pkl` files in the `Models/` directory.

## Results

| Model | Accuracy | Precision (0) | Recall (0) | Precision (1) | Recall (1) |
|---|---|---|---|---|---|
| Logistic Regression | 0.83 | 0.28 | 0.59 | 0.96 | 0.85 |
| Multinomial Naive Bayes | 0.91 | 0.50 | 0.03 | 0.92 | 1.00 |
| Linear SVC | 0.84 | 0.26 | 0.44 | 0.94 | 0.88 |

The dataset is heavily imbalanced toward positive reviews (class `1`), so overall accuracy is a misleading metric on its own. Naive Bayes has the highest accuracy but almost never detects negative reviews (recall of 0.03 for class 0), while Logistic Regression and Linear SVC trade some accuracy for meaningfully better recall on negative reviews — generally more useful for flagging complaints.

## Repository Structure

```
.
├── Sentiment_Analysis.ipynb     # Main notebook: scraping, EDA, preprocessing, modeling
├── opay_reviews_raw.csv         # Raw scraped review data (generated)
└── Models/
    ├── TfidfVectorize.pkl       # Fitted TF-IDF vectorizer
    ├── Logisticsregression.pkl  # Trained Logistic Regression model
    ├── naive.pkl                # Trained Naive Bayes model
    └── Linearsvc.pkl            # Trained Linear SVC model
```

## Requirements

- Python 3.13
- pandas, numpy
- matplotlib, seaborn, wordcloud
- nltk
- scikit-learn
- xgboost
- google-play-scraper

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn xgboost google-play-scraper
```

You'll also need to download the required NLTK corpora once:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
```

## Usage

1. Open `Sentiment_Analysis.ipynb` in Jupyter.
2. Run the cells in order — this will:
   - Scrape fresh review data (or you can use an existing `opay_reviews_raw.csv`).
   - Clean the data and run the EDA.
   - Preprocess text and extract TF-IDF features.
   - Train and evaluate the three models.
   - Save the trained models and vectorizer to `Models/`.

### Predicting on new text

```python
import pickle

tfidf = pickle.load(open("Models/TfidfVectorize.pkl", "rb"))
model = pickle.load(open("Models/Linearsvc.pkl", "rb"))  # or Logisticsregression.pkl / naive.pkl

review = "Transfers are fast and the app is easy to use."
clean = preprocess_text(review)  # defined in the notebook
vector = tfidf.transform([clean])
prediction = model.predict(vector)[0]  # 1 = positive, 0 = negative
```


## License

Specify a license for this project (e.g., MIT).
