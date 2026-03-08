import os
import json
import requests
from datetime import datetime, timedelta

HISTORY_DIR = "news_storage"

def get_local_sentiment(symbol_name):
    search_term = symbol_name.split(' ')[0].strip()
    safe_name = "".join([c for c in search_term if c.isalnum()])
    file_path = os.path.join(HISTORY_DIR, f"{safe_name}_history.json")

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                all_news = json.load(f)
                if not all_news: return 0.0, 0.0, "NEUTRAL", 0
                now = datetime.now()
                day_ago = now - timedelta(days=1)
                recent_news = [i for i in all_news if datetime.strptime(i['date_added'], "%Y-%m-%d %H:%M") > day_ago]
                avg_long = sum(i['score'] for i in all_news) / len(all_news)
                avg_short = sum(i['score'] for i in recent_news) / len(recent_news) if recent_news else avg_long
                status = "BULLISH" if avg_short > 0.1 else "BEARISH" if avg_short < -0.1 else "NEUTRAL"
                return avg_short, avg_long, status, len(recent_news)
            except: pass
    return 0.0, 0.0, "N/A", 0

def search_ticker_live(query):
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5).json()
        return [f"{q.get('symbol')} | {q.get('shortname', 'Asset')}" for q in response.get('quotes', [])]
    except: return []