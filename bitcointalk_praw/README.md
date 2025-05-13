# Bitcointalk BTC Price Scraper (Simplified Version)

## Project Overview

This project implements a simple Bitcointalk forum scraper focused on Bitcoin-related discussions. It starts from a specified topic ID and crawls backward (decrementing the ID) to collect posts within a specific date range and matching specified keywords.

## Features

- Logs in to the Bitcointalk forum using Selenium
- Starts from a given topic ID and decrements one by one
- Parses each topic and filters by:
  - Date range (start date to end date)
  - Presence of target keywords in title or content
- Saves valid posts as JSON files to `data/json/`
- Supports resuming from last state using `checkpoint.json`
- Applies random sleep between requests to reduce server load
- Logs progress and errors to the console or a log file

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running the scraper

```bash
python bitcointalk_scraper.py
```

```

## Data Output

Each valid post is saved in the `data/json/` directory as:

```
topic_<topic_id>.json
```

The scraper also creates a `checkpoint.json` file to allow resuming after interruptions.

## Limitations

This simplified scraper does not include:

- CAPTCHA automation or detection
- Session reinitialization on failure
- Board-page or pagination scraping
- Comment vote parsing or advanced interaction metrics
- Retry or backoff mechanisms on network errors

## Suitable For

- Small to medium-scale historical data extraction
- Keyword-based topic filtering
- Time-bounded topic collection by topic ID

For more advanced scraping scenarios (e.g., full board traversal, CAPTCHA handling, session resilience), refer to the full version of the scraper (not used in this project).