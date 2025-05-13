# Reddit Bitcoin Post Crawling Tool

This is a Python script project that uses the official Reddit API to scrape posts from the r/Bitcoin subreddit and save the data in JSON format. It provides three scraping scripts: a basic version, an advanced version, and a date range version.

## Features

- Logs in using Reddit's OAuth2 authorization method
- Scrapes as many historical posts from r/Bitcoin as possible, starting from the latest posts
- Supports scraping various information: title, author, publication time, score, number of comments, URL, body text, etc.
- Uses a pagination mechanism to loop through historical data
- Supports filtering posts by date range
- Supports batch saving of data to prevent individual files from becoming too large
- Built-in request delays and error handling to avoid triggering API limits
- Complete logging and progress display

## Project Structure

```
reddit-bitcoin-scraper/
├── reddit_bitcoin_scraper.py         # Basic version scraping script
├── reddit_bitcoin_scraper_advanced.py    # Advanced version scraping script
├── reddit_bitcoin_scraper_by_date.py     # Date range scraping script
├── requirements.txt                  # Project dependencies
├── .env.example                      # Example environment variable file
└── README.md                         # Project documentation
```

## Dependencies

This project depends on the following Python libraries:

- praw (Python Reddit API Wrapper)
- requests
- python-dotenv
- tqdm

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Obtaining Reddit API Credentials

Before using the scripts, you need to obtain access credentials for the Reddit API:

1. Log in to your Reddit account
2. Go to https://www.reddit.com/prefs/apps
3. Click the "create app" or "create another app" button at the bottom of the page
4. Fill in the application information:
   - Choose "script" as the type
   - Name: Custom, e.g., "Bitcoin Scraper"
   - Description: Optional
   - About URL: Optional
   - Redirect URI: Enter "http://localhost:8080"
5. Click the "create app" button to create the application
6. After successful creation, record the following information:
   - client_id: A 14-character string under the application name
   - client_secret: The string displayed as "secret"

## Configuration File

Copy the `.env.example` file and rename it to `.env`, then edit the file to fill in your API credentials:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT="r/Bitcoin Scraper by /u/YOUR_REDDIT_USERNAME"
```

## Usage

### 1. Basic Version Scraping Script

```bash
python reddit_bitcoin_scraper.py [--config .env] [--output data] [--batch-size 1000] [--delay 1.0] [--limit <max_posts>]
```

Parameter descriptions:
- `--config`: Path to the configuration file, default is .env
- `--output`: Output directory, default is data
- `--batch-size`: Number of posts saved per batch, default is 1000
- `--delay`: Request interval time (seconds), default is 1.0 seconds
- `--limit`: Maximum number of posts to scrape, default is unlimited

### 2. Advanced Version Scraping Script

```bash
python reddit_bitcoin_scraper_advanced.py [--config .env] [--output data] [--batch-size 1000] [--delay 1.0] [--limit-per-request 100] [--max-requests <number>] [--sort new|hot|top]
```

Parameter descriptions:
- `--config`: Path to the configuration file, default is .env
- `--output`: Output directory, default is data
- `--batch-size`: Number of posts saved per batch, default is 1000
- `--delay`: Request interval time (seconds), default is 1.0 seconds
- `--limit-per-request`: Number of posts per request, default is 100
- `--max-requests`: Maximum number of requests, default is unlimited
- `--sort`: Post sorting method, options are new/hot/top, default is new

### 3. Date Range Scraping Script

```bash
python reddit_bitcoin_scraper_by_date.py [--config .env] [--output data] [--batch-size 1000] [--delay 1.0] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--days-ago <days>] [--limit-per-request 100] [--max-requests <number>] [--sort new|hot|top]
```

Parameter descriptions:
- `--config`: Path to the configuration file, default is .env
- `--output`: Output directory, default is data
- `--batch-size`: Number of posts saved per batch, default is 1000
- `--delay`: Request interval time (seconds), default is 1.0 seconds
- `--start-date`: Start date, format is YYYY-MM-DD
- `--end-date`: End date, format is YYYY-MM-DD
- `--days-ago`: Scrape posts from how many days ago
- `--limit-per-request`: Number of posts per request, default is 100
- `--max-requests`: Maximum number of requests, default is unlimited
- `--sort`: Post sorting method, options are new/hot/top, default is new

**Note:** If both `--start-date` and `--days-ago` are specified, `--start-date` will take precedence.

## Output Data

The scripts will save post data as JSON files in the specified output directory (default is the "data" directory). Each batch of data is saved as a separate JSON file, with the filename format:

```
bitcoin_posts_batch_<batch_number>_<timestamp>.json
```

For the date range scraping script, the filename format is:

```
bitcoin_posts_<date_range>_batch_<batch_number>_<timestamp>.json
```

Each post data includes the following fields:

- id: Post ID
- title: Title
- author: Author
- created_utc: Publication time (Unix timestamp)
- created_time: Publication time (readable format)
- score: Score (upvote minus downvote)
- upvote_ratio: Approval rate
- num_comments: Number of comments
- permalink: Relative URL on Reddit
- url: Post link
- is_self: Whether it is a self-post
- selftext: Body content
- stickied: Whether it is pinned
- over_18: Whether it is marked as adult content
- spoiler: Whether it contains spoilers
- link_flair_text: Link flair text

## Usage Examples

### Scrape posts from the last week

```bash
python reddit_bitcoin_scraper_by_date.py --days-ago 7
```

### Scrape posts from January 1, 2023, to June 30, 2023

```bash
python reddit_bitcoin_scraper_by_date.py --start-date 2023-01-01 --end-date 2023-06-30
```

### Scrape posts from January 1, 2023, to now, with a maximum of 50 posts per request and a 2-second interval

```bash
python reddit_bitcoin_scraper_by_date.py --start-date 2023-01-01 --limit-per-request 50 --delay 2.0
```

## Notes

1. The Reddit API has access frequency limits; please do not set too short request delays to avoid temporary bans.
2. Scraping a large amount of historical data may take a long time; it is recommended to use the advanced version script or the date range script.
3. If you encounter API errors, please check the log file for detailed information.
4. It is recommended to run this script using Python 3.7 or higher.
5. The date range scraping script can more effectively obtain data for specific time periods, reducing unnecessary requests.

## License

MIT

## Disclaimer

This project is for learning and research purposes only. When using this tool, please comply with Reddit's terms of service and API usage policies. The author is not responsible for any issues arising from the use of this tool.
