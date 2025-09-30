# IMPORT LIBERARIES
import requests
import pandas as pd
import numpy as np
from datetime import datetime 
import re
import json
import re
import time
from datetime import datetime
from urllib.parse import urlencode
from typing import List, Dict, Set
import requests
import pandas as pd # <-- NEW: Added pandas for DataFrame output
import csv
import json
import time
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ------------------------------------------------------------------------------------------------------------------------------------------
# WEBSITE HEADER DEFINE FOR DATA PRICE INGESTION 
yahoo_finance_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Cookie': 'A1=d=AQABBJkNsWgCEEi_58VJbGB_l7OJZ7XWrc4FEgEBAQFfsmi6aK8AAAAA_eMCAA&S=AQAAAl61G0LkQqr0GzwTZZlMD30; A3=d=AQABBJkNsWgCEEi_58VJbGB_l7OJZ7XWrc4FEgEBAQFfsmi6aK8AAAAA_eMCAA&S=AQAAAl61G0LkQqr0GzwTZZlMD30; A1S=d=AQABBJkNsWgCEEi_58VJbGB_l7OJZ7XWrc4FEgEBAQFfsmi6aK8AAAAA_eMCAA&S=AQAAAl61G0LkQqr0GzwTZZlMD30; PRF=t%3DTSLA; gpp=DBAA; gpp_sid=-1; fes-ds-session=pv%3D7; cmp=t=1756438244&j=0&u=1---',
        'Priority': 'u=0, i',
        'Sec-Ch-Ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36'
    }

# WEBSITE CONFIGS FOR NEWS INGESTION 
BASE_URL = "https://api.queryly.com/json.aspx"
JSONP_REGEX = re.compile(r"searchPage\.resultcallback\((.*)\)", re.DOTALL)
COMMON_PARAMS = {
    "queryly_key": "459f578c8fd040d1",
    "callback": "searchPage.resultcallback",
    "showfaceted": "true",
    "extendeddatafields": "creator,_media,subheader",
    "timezoneoffset": "-450",
}
BATCH_SIZE = 500

# WEBSITE CONFIGS FOR NEWS CONTENT INGESTION  
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
)
# ------------------------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS 
def _parse_jsonp(text: str) -> dict:
    """Strips the JSONP callback wrapper to extract and return the raw JSON dictionary."""
    match = JSONP_REGEX.search(text.strip())
    if not match:
        raise ValueError("Could not find JSONP callback in the response.")
    json_text = match.group(1).strip()
    # Simple logic to handle JSONP wrapping (uses the core logic from the original)
    open_braces = 0
    end_index = -1
    for i, char in enumerate(json_text):
        if char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
        if open_braces == 0 and i > 0:
            end_index = i + 1
            break
    if end_index != -1:
        json_text = json_text[:end_index]  
    return json.loads(json_text)

def _to_iso_date(human_date: str) -> str:
    """Converts a human-readable date string (e.g., 'Sep 05, 2025') to ISO format."""
    if not human_date:
        return ""
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(human_date, fmt).date().isoformat()
        except ValueError:
            pass
    return human_date

def _fetch_batch(query: str, endindex: int, session: requests.Session) -> dict:
    """Fetches a single batch of results for a given query."""
    params = {
        **COMMON_PARAMS,
        "query": query,
        "endindex": str(endindex),
        "batchsize": str(BATCH_SIZE),
    }
    url = f"{BASE_URL}?{urlencode(params)}"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return _parse_jsonp(resp.text)
    except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
        print(f"Failed to fetch or parse batch for query '{query}' at index {endindex}: {e}")
        return {}

def _fetch_html(session: requests.Session, url: str) -> Optional[str]:
    """
    Fetches HTML content for a given URL using the provided session.
    """
    try:
        # Added a short delay before fetching to mimic human behavior
        time.sleep(0.2)
        resp = session.get(url, timeout=25)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [!] Failed to fetch {url}: {e}")
        return None

def _find_newsarticle_jsonld(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """
    Finds and parses the NewsArticle JSON-LD script tag from HTML.
    """
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            # The data can be a single dict or a list of dicts
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and item.get("@type") in ("NewsArticle", "Article"):
                    return item
        except (json.JSONDecodeError, TypeError):
            continue
    return None

def _clean_author(author_data: Any) -> str:
    """
    Extracts and cleans author names from various JSON-LD formats.
    """
    if not author_data:
        return ""
    if isinstance(author_data, str):
        return author_data.strip()
    if isinstance(author_data, dict):
        return (author_data.get("name") or "").strip()
    if isinstance(author_data, list):
        names = []
        for item in author_data:
            name = _clean_author(item)
            if name:
                names.append(name)
        return ", ".join(names)
    return ""

# ------------------------------------------------------------------------------------------------------------------------------------------
# DEFINE FUNCTION TO IMPORT STOCK PRICE 
def us_stock_price_yahoo_finance(ticker = 'TSLA', period1 = '1262204200', period2='1756433964', ingest_header = yahoo_finance_headers):
    # Send requests 
    response = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?&includeAdjustedClose=true&interval=1d&period1={period1}&period2={period2}",
                           headers = ingest_header)
    data = response.json()
    # navigate to the data
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quotes = result['indicators']['quote'][0]
    adjclose = result['indicators']['adjclose'][0]['adjclose']
    
    # build dataframe
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='s'),
        'open': quotes['open'],
        'high': quotes['high'],
        'low': quotes['low'],
        'close': quotes['close'],
        'adjclose': adjclose,
        'volume': quotes['volume']
    })

    # add custom date fields
    df['datetime'] = df['timestamp'].dt.strftime('%Y%m%d')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['ticker'] = ticker
    df['table_name'] = 'us_stock_price_yahoo_finance'
    
    return(df)

# --------------------------------------------------------------------------------------------------
# # DEFINE FUNCTION TO CRAWL NEWS LINKS, CONTENTS 
def nbc_news_article_crawler(queries: List[str]) -> pd.DataFrame:
    """
    Crawl article links from the NBC News search API for a list of queries,
    handles pagination, deduplication, and returns results as a pandas DataFrame.
    """
    all_rows: List[Dict[str, str]] = []
    seen_links: Set[str] = set()
    session = requests.Session()
    
    for query in queries:
        print(f"\n--- Starting crawl for query: '{query}' ---")
        endindex = 0
        while True:
            data = _fetch_batch(query, endindex, session)
            items = data.get("items", [])
            
            if not items:
                print(f"No more items found for '{query}'. Moving to next query.")
                break

            total = data.get("metadata", {}).get("total", "N/A")
            print(f"Fetched batch starting at {endindex}: {len(items)} items (total found: {total})")

            new_articles_found = 0
            for item in items:
                if item.get("feedname") != "articles":
                    continue
                
                link = (item.get("link") or "").strip()
                if not link or link in seen_links:
                    continue
                
                seen_links.add(link)
                
                title = (item.get("title") or "").strip()
                date_iso = _to_iso_date(item.get("pubdate", "").strip())
                
                all_rows.append({
                    "title": title,
                    "date": date_iso,
                    "link": link,
                    "query": query # Add the search query for context
                })
                new_articles_found += 1
            
            print(f"Added {new_articles_found} new, unique articles.")
            
            endindex += len(items)
            time.sleep(0.25) # Be respectful to the server

    print(f"\nCompleted crawl. Total unique articles found: {len(all_rows)}")
    
    # CONVERT TO DATAFRAME
    if not all_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rows)
    
    # Sort the DataFrame by date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(by='date', ascending=False).reset_index(drop=True)
    
    return df


def scrape_nbc_articles(input_csv: str, limit: int = 10000) -> Optional[pd.DataFrame]:
    """
    Reads links from the input CSV, scrapes the content for each article
    up to the specified limit, and returns the data as a Pandas DataFrame.

    Args:
        input_csv (str): Path to the CSV file containing article links.
        limit (int): The maximum number of articles to scrape.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the scraped data, or None.
    """
    scraped_articles: List[Dict[str, str]] = []
    
    # Setup session
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Read CSV and apply limit
    try:
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            articles_to_scrape = list(reader)[:limit]
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv}'")
        return None
    print(f"Found {len(articles_to_scrape)} articles (showing first {len(articles_to_scrape)}) to scrape from '{input_csv}'.")

    # Loop and scrape
    for i, article_info in enumerate(articles_to_scrape):
        url = article_info.get("link")
        if not url:
            continue
        print(f"({i+1}/{len(articles_to_scrape)}) Scraping: {url}")
        
        # Fetch and parse
        html = _fetch_html(session, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        json_ld = _find_newsarticle_jsonld(soup)
        if not json_ld:
            print(f"  [!] No NewsArticle JSON-LD found for {url}. Skipping.")
            continue

        # Extract data
        title = (json_ld.get("headline") or article_info.get("title", "")).strip()
        date_published = (json_ld.get("datePublished") or article_info.get("date", "")).strip()
        author = _clean_author(json_ld.get("author"))
        content = (json_ld.get("articleBody") or "").strip()

        scraped_articles.append({
            "title": title,
            "date": date_published,
            "link": url,
            "author": author,
            "content": content,
        })
        
    if not scraped_articles:
        print("No data was successfully scraped.")
        return None

    # 6. Convert to DataFrame and sort
    df = pd.DataFrame(scraped_articles)
    
    if "date" in df.columns:
        df = df.sort_values(by="date", ascending=False)
    
    print(f"\nSuccessfully scraped and created a DataFrame with {len(df)} articles.")
    return df
