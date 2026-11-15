# Project: BNPL in Saudi Arabia — What Users Are Saying

## Goal
A portfolio data science project analyzing public user reviews of **Tabby** and **Tamara**, Saudi Arabia's two dominant Buy Now, Pay Later (BNPL) apps. It covers data collection, Arabic NLP, time-series analysis, a served model, and a live Flutter web dashboard.

Main questions:
1. How do users feel about Tabby vs. Tamara, and how has that changed over time?
2. What do users complain about and praise most (fees, rejections, credit limits, customer service, app bugs, merchant issues)?
3. Do sentiment shifts line up with real events (SAMA regulation updates, limit increases, Ramadan / White Friday seasons)?
4. How well can a model classify Arabic BNPL reviews by sentiment and topic, and does a pretrained Arabic transformer beat a classical baseline?

## Owner context
- Owner: Omar (data science student, University of Jeddah). Strong in Flutter + Firebase.
- Goal: a GitHub portfolio piece for co-op applications. Code must be clean, reproducible and well documented.
- Write complete files, not partial snippets.

## Tech stack
- Python 3.11+, pandas, numpy, scikit-learn, matplotlib/plotly
- Data collection: `google-play-scraper` (PyPI)
- Arabic NLP: Hugging Face `transformers` (e.g. `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`), `camel-tools` or simple custom normalization
- Topic modeling: `BERTopic` with a multilingual sentence-transformer (e.g. `paraphrase-multilingual-MiniLM-L12-v2`)
- API: FastAPI + uvicorn, Dockerfile for Hugging Face Spaces (free CPU)
- Front end: Flutter web, deployed on Firebase Hosting
- Data storage: Parquet files in `data/` (raw data is NOT committed; see Data rules)

## Repository structure
```
bnpl-saudi-reviews/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml              # app IDs, date ranges, model names, paths
├── data/
│   ├── raw/                 # gitignored
│   ├── processed/           # gitignored except small samples
│   ├── labeled/             # hand-labeled gold set (anonymized, committed)
│   └── external/            # Google Trends CSVs, events timeline
├── notebooks/
│   ├── 01_collection_check.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_sentiment_models.ipynb
│   ├── 04_topics.ipynb
│   └── 05_trends_and_events.ipynb
├── src/
│   ├── collect/
│   │   ├── play_store.py    # scrape Google Play reviews for both apps
│   │   └── app_store.py     # optional: Apple public RSS reviews (recent only)
│   ├── clean/
│   │   ├── normalize.py     # Arabic normalization, emoji handling, dedup
│   │   └── language.py      # detect Arabic / English / mixed
│   ├── models/
│   │   ├── baseline.py      # TF-IDF + Logistic Regression
│   │   ├── transformer.py   # CAMeLBERT inference / optional fine-tune
│   │   ├── topics.py        # BERTopic
│   │   └── evaluate.py      # metrics on gold set
│   └── analysis/
│       └── trends.py        # monthly aggregates, event overlay
├── api/
│   ├── main.py              # FastAPI: /predict, /stats, /topics, /trends
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/               # Flutter web app
├── reports/figures/         # exported charts used in README
└── tests/
```

## Phases

### Phase 1: Data collection
- Find the correct Google Play package IDs for Tabby and Tamara from their Play Store URLs and store them in `config.yaml`. Do not guess IDs.
- Collect all available reviews in country `sa`, language `ar` and also `en`. Keep: review id, app, text, star rating, date, thumbs-up count, app version, developer reply (yes/no + date).
- **Drop reviewer names and profile images at collection time.** Never store them.
- Save to `data/raw/play_reviews_<app>_<date>.parquet`. Script must be re-runnable and incremental (only fetch new reviews).
- Optional: Apple App Store public RSS feed (limited to recent reviews). Label the source clearly.
- Google Trends: download CSVs manually from trends.google.com for "تابي", "تمارا", "Tabby", "Tamara" (Saudi Arabia, 2020 to present) into `data/external/`. Don't rely on unofficial Trends scrapers.
- `data/external/events.csv`: a hand-made timeline of key events (SAMA BNPL rules, licenses, limit changes, funding rounds, Ramadan / White Friday dates) with date, event, source URL.

### Phase 2: Cleaning and EDA
- Arabic normalization: unify alef forms, taa marbuta / haa, remove tatweel and diacritics, handle repeated letters ("حلووووو") and emojis (keep emoji as a feature).
- Language tag: Arabic / English / mixed / Arabizi.
- Remove exact duplicates and empty or very short reviews; log how many were removed and why.
- EDA: review volume per month per app, rating distribution, review length, language mix, developer reply rate and response time, top words / bigrams per rating.

### Phase 3: Sentiment modeling
- Weak labels from stars: 1–2 = negative, 3 = neutral, 4–5 = positive.
- **Gold test set:** Omar hand-labels a random sample of about 300–500 reviews (sentiment + main topic) in `data/labeled/gold.csv`. Build a small labeling helper (script or notebook) to make this fast. All final metrics are reported on this gold set, not on star labels.
- Models to compare:
  1. Baseline: TF-IDF (char n-grams work well for Arabic) + Logistic Regression trained on star labels.
  2. Pretrained CAMeLBERT sentiment model, zero-shot.
  3. Optional: CAMeLBERT fine-tuned on star labels (Colab GPU if needed).
- Report macro-F1, per-class precision/recall and a confusion matrix. Include an error analysis: examples where the model fails (sarcasm, dialect, mixed language).

### Phase 4: Topic modeling
- BERTopic on negative and positive reviews separately.
- Merge and name topics into a readable set (e.g. Fees & late payment, Rejection / limit, Customer service, App bugs, Merchant / refund issues, Ease of use).
- Topic share per app and per month.

### Phase 5: Trends and events
- Monthly sentiment index per app (share positive minus share negative), with confidence intervals.
- Overlay `events.csv` and Google Trends. Describe what lines up, and be careful to say these are correlations, not causes.

### Phase 6: API
- FastAPI endpoints:
  - `POST /predict` {text} → sentiment, probabilities, topic
  - `GET /stats` → per-app summary numbers
  - `GET /trends?app=` → monthly sentiment series
  - `GET /topics?app=` → topic shares
- Precompute aggregates into JSON at build time; only `/predict` runs the model.
- CORS enabled for the dashboard domain. Dockerfile ready for Hugging Face Spaces.

### Phase 7: Flutter web dashboard
- Pages: Overview (key numbers, Tabby vs. Tamara), Sentiment over time (with events), Topics, Try the model (text box → prediction), About / Method.
- Arabic and English UI (RTL support).
- Deploy to Firebase Hosting. Link it in the README.

### Phase 8: README and polish
- README sections: one-line summary, live demo link + screenshot, key findings (3–5 bullets with numbers), data and method, model results table, limitations, how to run, project structure, ethics note.
- Limitations to state honestly: app reviews are not a random sample of users, unhappy users review more, star labels are noisy, no access to company transaction data.
- Ethics note: only public review text and ratings are used, no names or personal data are stored or published, raw data is not redistributed, results describe user opinions, not company performance.

## Data rules
- Never commit raw scraped data. Commit only: the anonymized gold set, small anonymized samples (≤200 rows), aggregates, and figures.
- Never store reviewer names, profile pictures or user IDs.
- Keep collection polite: rate-limit requests.

## Coding standards
- Type hints and docstrings on all functions; config values come from `config.yaml`, no hard-coded paths.
- Set random seeds everywhere for reproducibility.
- Notebooks call functions from `src/`; heavy logic lives in `src/`, not in notebooks.
- Each phase ends with a short entry in `README.md` or `reports/` and a clean commit.
