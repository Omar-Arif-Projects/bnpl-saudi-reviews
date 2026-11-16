# BNPL in Saudi Arabia: What Users Are Saying

An analysis of 184,000 public app-store reviews of **Tabby** and **Tamara**, Saudi Arabia's two largest Buy Now, Pay Later apps, using Arabic and English NLP.

> **Status:** data collection, cleaning and exploratory analysis are complete. Sentiment models, topic modeling, the events analysis, an API and a web dashboard are in progress (see [Roadmap](#roadmap)).

## Key findings

From 183,341 Google Play reviews (Saudi Arabia, October 2020 to November 2026):

- **Ratings are very positive.** 85% of reviews are 5 stars. Positive reviews are short (median 2 words), while 1-star reviews have a median of 12–13 words, so most of the detail about problems is in the negative reviews.
- **Tamara's review mix changed in February 2026.** Positive reviews fell by about 90% (from about 2,200 to 190 a month), negative reviews by about 35% (from 113 to 72). The average rating dropped from 4.7 to 3.9 mainly because of a change in who reviews, not a rise in complaints.
- **Tabby's negative reviews spiked in July–August 2026**, reaching 371 and 562 a month against a usual 150–230.
- **Both companies reply to about 90% of 1–2 star reviews.** The difference in overall reply rate (Tabby 56%, Tamara 11%) comes from Tabby also replying to positive reviews, which it has done since 2024. Tamara replies faster (median under 1 hour vs. about 22 hours).
- **The most frequent complaints** are customer service (both apps), app updates showing a blank white screen (Tabby), and late fees and the first payment (Tamara).

![Positive and negative reviews per month](reports/figures/eda_reviews_by_sentiment.png)

![Distinctive two-word phrases in Arabic reviews](reports/figures/eda_terms_arabic.png)

More charts and tables are in [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb).

## Data

| Source | Coverage | Use |
|---|---|---|
| Google Play reviews (`google-play-scraper`): `app.tabby.client`, `co.tamara.user` | Full available history, country `sa`, Arabic and English queries; 183,341 reviews after cleaning | Main dataset |
| Apple App Store public RSS: `1528022364`, `1533512531` | Most recent 500 reviews per app | Comparison sample |
| Google Trends, manual export | Monthly search interest for "تابي" and "تمارا" in Saudi Arabia from December 2018, plus interest by region | Context for the events analysis |
| SAMA payments statistics via [KAPSARC](https://data.kapsarc.org) (`pos-transactions`) | Monthly POS and mada e-commerce sales and transactions from January 2019 | Context for the events analysis |
| Events timeline (`data/external/events.csv`) | Regulation, funding, product and seasonal events, each with a source URL | Events analysis (in progress) |

**Fields kept:** review ID, app, source, query language, text, star rating, review date, thumbs-up count, app version, whether the developer replied, and the reply date. Reviewer names, profile images and user IDs are never stored.

**Notes on the data:**
- Package IDs were checked against the live store listings. Tabby's `app.tabby.cashier` is a separate merchant app and is not used.
- The Google Trends values are a relative index (0–100) on a common scale for both terms, not search volumes. The English terms "Tabby" and "Tamara" were excluded because they had much lower search interest and are ambiguous. Export settings are recorded in `config.yaml`.
- The SAMA e-commerce series covers mada debit cards only and excludes Visa, Mastercard and credit-card spending.

![Google search interest for Tabby and Tamara in Saudi Arabia](reports/figures/trends_interest_over_time.png)

## Method

**Collection.** Reviews are collected incrementally: each run fetches only reviews newer than the last stored date, with a one-day overlap, rate limiting and retries. Raw files are stored as Parquet.

**Cleaning.** The pipeline ([`src/clean/pipeline.py`](src/clean/pipeline.py)) removed 2 double submissions and 176 reviews with no usable content. Identical text from different reviewers (for example "جيد" or "👍") is kept, since each is a separate opinion. Each step is recorded in [`reports/cleaning_log.csv`](reports/cleaning_log.csv).

**Arabic normalization.** A normalized copy of each review is stored next to the original text. It applies Unicode NFKC, unifies alef forms (أ إ آ → ا), taa marbuta (ة → ه) and alef maqsura (ى → ي), removes diacritics and tatweel, converts Arabic-Indic digits, and caps repeated letters at two (حلووووو → حلوو). Emoji are extracted into a separate column for use as features.

**Language detection.** Language is detected from the text rather than from the store's query language, because each query bucket contains about 2,000 reviews in the other language. Reviews are tagged as Arabic (50%), English (47%), emoji only (2.6%) or mixed (0.1%). No Arabizi (Arabic written in Latin letters) was found. Emoji-only reviews are included in the analysis but excluded from text-model training.

**Gold test set.** 400 Google Play reviews are labeled by hand for sentiment and main topic, with the star rating hidden. The sample is stratified by app, language and star group (40% 1–2 star, 25% 3 star, 35% 4–5 star per app and language), so negative and neutral reviews are over-sampled; each row carries a weight that restores population proportions. Review IDs are replaced by a hash, and e-mail addresses, URLs and long numbers are masked. Gold reviews are excluded from model training.

**Distinctive phrases.** Words and two-word phrases are ranked with the weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008), comparing 1–2 star with 4–5 star reviews, separately for each app and language.

## Limitations

- App reviews are not a random sample of users. Unhappy users are more likely to write long reviews, and in-app rating prompts affect who reviews, as the Tamara change in February 2026 shows.
- Star ratings are a noisy measure of sentiment. Final model results will be reported on a hand-labeled test set.
- Review dates reflect the latest edit of a review, and reply dates the latest edit of a reply, so response times are upper bounds.
- "تمارا" is also a given name, so part of Tamara's search interest is unrelated to the app.
- There is no access to company transaction data. Results describe what users write, not how the companies perform.

## Project structure

```
├── config.yaml              # app IDs, paths and parameters
├── src/
│   ├── collect/             # Google Play, App Store RSS, KAPSARC, Google Trends
│   ├── clean/               # normalization, language detection, cleaning pipeline
│   ├── label/               # gold-set sampling and labeling widget
│   └── analysis/            # EDA tables and chart helpers
├── notebooks/               # collection checks, EDA and gold-set labeling
├── reports/                 # cleaning log and figures
├── data/                    # raw and processed data (not committed), external sources
└── tests/
```

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.collect.play_store                   # Google Play reviews, both apps (incremental)
python -m src.collect.play_store --max-reviews 50  # quick test
python -m src.collect.app_store                    # App Store RSS reviews
python -m src.collect.sama_kapsarc                 # SAMA payments statistics
python -m src.collect.google_trends                # clean Google Trends exports
python -m src.clean.pipeline                       # clean and normalize reviews
jupyter notebook notebooks/02_eda.ipynb            # exploratory analysis
python -m src.label.gold --sample                  # draw the gold-set sample
jupyter notebook notebooks/gold_labeling.ipynb     # label it

pytest
```

All settings (app IDs, country, languages, rate limits, paths, cleaning thresholds) are in `config.yaml`.

## Roadmap

- [x] Data collection
- [x] Cleaning, normalization and exploratory analysis
- [ ] Sentiment models: TF-IDF baseline vs. a pretrained Arabic transformer, evaluated on a hand-labeled test set
- [ ] Topic modeling of complaints and praise
- [ ] Sentiment trends compared with regulation, product and seasonal events
- [ ] Prediction API (FastAPI)
- [ ] Web dashboard (Flutter)

## Data and ethics

Only public review text and ratings are used. No names or personal data are stored or published, and raw review data is not committed or redistributed (`data/raw/` and `data/processed/` are gitignored). The repository contains only code, aggregate tables, figures and, later, an anonymized hand-labeled test set. Results describe user opinions, not company performance.
