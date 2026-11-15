# BNPL in Saudi Arabia: What Users Are Saying

An analysis of public app-store reviews of **Tabby** and **Tamara**, Saudi Arabia's two biggest Buy Now, Pay Later apps. It covers data collection, Arabic NLP, sentiment and topic modeling, trends over time against real events, a served model, and a Flutter web dashboard.

> 🚧 Work in progress, built phase by phase. Live demo, key findings and model results will appear here.

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Data collection | ✅ code done, data collected locally |
| 2 | Cleaning and EDA | ⏳ |
| 3 | Sentiment modeling | ⏳ |
| 4 | Topic modeling | ⏳ |
| 5 | Trends and events | ⏳ |
| 6 | API | ⏳ |
| 7 | Flutter dashboard | ⏳ |
| 8 | README and polish | ⏳ |

## Phase 1: Data collection

**Sources**

| Source | App IDs | Coverage | Notes |
|---|---|---|---|
| Google Play (`google-play-scraper`) | `app.tabby.client`, `co.tamara.user` | Full available history, country `sa`, languages `ar` + `en` | Main dataset |
| Apple App Store public RSS | `1528022364`, `1533512531` | Most recent reviews only (max 500 per app) | Optional, labelled `source = app_store_rss`; not comparable in volume. In Nov 2026 the SA feed returned no reviews for Tabby |
| Google Trends (manual CSV export) | "تابي", "تمارا", "Tabby", "Tamara" | Saudi Arabia, 2020 to present | Downloaded by hand from trends.google.com into `data/external/` |
| SAMA payments statistics via [KAPSARC](https://data.kapsarc.org) | dataset `pos-transactions` | Monthly, Jan 2019 onward in the wide file | POS and mada e-commerce sales and transactions, as macro context for BNPL growth |
| Events timeline | n/a | n/a | Hand-made `data/external/events.csv` with a source URL for every row |

Package IDs were checked against the live store listings (Tabby's `app.tabby.cashier` is the separate *Tabby Business* app, so it is not used).

**Fields kept:** review id, app, source, country, query language, text, star rating, review date, thumbs-up count, app version, whether the developer replied, and the reply date.
**Never stored:** reviewer names, profile images, user IDs, or developer reply text.

**SAMA payments data:** `python -m src.collect.sama_kapsarc` saves the full dataset (`sama_pos_ecommerce.csv`, long format) and a monthly wide file (`sama_pos_ecommerce_monthly_wide.csv`) starting January 2019. In the wide file, sales are in millions of SAR, columns have short names (e.g. `ecom_mada_sales_m_sar`, `ecom_mada_txn_count`, `pos_sales_m_sar`), and there are two derived columns: `ecom_avg_ticket_sar` (e-commerce sales / transactions) and `ecom_share_of_card_spending` (e-commerce / (e-commerce + POS) sales).
> **Note:** the e-commerce series covers **mada debit cards only**. It excludes Visa, Mastercard and credit-card online spending, so it understates total online spending.

**Incremental collection:** each run reads what is already in `data/raw/` and only pages through reviews newer than the last stored date (with a one-day overlap). New reviews go to `data/raw/play_reviews_<app>_<YYYYMMDD>.parquet`. Requests are rate-limited (`collection.play_store.sleep_seconds` in `config.yaml`) and retried with backoff.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.collect.play_store                 # Google Play, both apps (incremental)
python -m src.collect.play_store --max-reviews 50  # quick test
python -m src.collect.play_store --full          # rescan full history, skipping stored IDs
python -m src.collect.app_store                  # optional Apple RSS reviews
python -m src.collect.sama_kapsarc               # SAMA POS / mada e-commerce statistics

pytest
```

All settings (app IDs, country, languages, rate limits, paths) live in `config.yaml`.

## Data and ethics

Raw scraped data is not committed or redistributed (`data/raw/` is gitignored). Only public review text and ratings are used, and no names or personal data are stored. The repository will contain only an anonymized hand-labelled gold set, small anonymized samples, aggregates and figures.
