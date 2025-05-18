import pandas as pd
from textblob import TextBlob
from datetime import datetime
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from groq import Groq
from groq.types.chat import ChatCompletionUserMessageParam  # ✅ FIXED import

# Initialize Groq client
client = Groq(api_key="Enter your grok api key.")  # Replace with your actual Groq API key


def generate_search_query(ticker):
    """
    Use Groq API to generate a UK-focused search query for stock news.
    """
    print("generating search query")
    try:
        prompt = (
            f"Generate a concise Google search query for recent stock news about ticker {ticker} "
            f"from UK-based sources (e.g., BBC, Financial Times, .co.uk domains). "
            f"Example: 'AAPL stock news site:*.co.uk'."
        )

        response = client.chat.completions.create(
            model="gemma-7b-it",
            messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],  # ✅ FIXED
            max_tokens=50,
            temperature=0.7
        )
        query = response.choices[0].message.content.strip()
        return query if query else f"{ticker} stock news site:*.co.uk"
    except Exception:
        return f"{ticker} stock news site:*.co.uk"


def extract_headline_and_date(url, ticker, start_date, end_date):
    print("extracting headline")
    """
    Scrape headline and date from a news URL, filter by date range and ticker.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Headline
        headline_tag = soup.find(['h1', 'h2', 'title']) or soup.find('meta', property='og:title')
        headline = (
            headline_tag.get('content') if hasattr(headline_tag, 'get') else headline_tag.text
        ).strip() if headline_tag else ""

        if not headline or ticker.lower() not in headline.lower():
            return None

        # Date
        date_tag = soup.find('meta', property='article:published_time') or soup.find('time')
        if date_tag:
            date_str = date_tag.get('content') if hasattr(date_tag, 'get') else date_tag.text
            try:
                pub_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
            except ValueError:
                return None

            if start_date <= pub_date <= end_date:
                return {'Date': pub_date.strftime('%Y-%m-%d'), 'Headline': headline}
    except Exception:
        pass
    return None


def get_sentiment_score(ticker, start_date, end_date):
    print("the function is called 1.")
    """
    Fetch stock news and compute sentiment scores.
    """
    try:
        print("the function is called 2.")
        if not ticker or not isinstance(ticker, str):
            print("the function is called 3.")
            return None, None, "Invalid ticker symbol"

        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            print("the function is called 3.")
        except ValueError:
            print(start_date, end_date)
            print("the function is called 5.")
            return None, None, "Invalid date format: Use YYYY-MM-DD"

        # Generate search query using Groq
        query = generate_search_query(ticker)

        # Fetch news URLs
        try:
            print("the function is called 6.")
            news_urls = list(search(query, num_results=10, lang="en"))
        except Exception as e:
            print("the function is called 7.")
            return None, None, f"Google Search error: {str(e)}"

        # Process articles
        sentiment_data = []
        for url in news_urls:
            result = extract_headline_and_date(url, ticker, start, end)
            if result:
                sentiment = TextBlob(result['Headline']).sentiment.polarity
                sentiment_data.append({'Date': result['Date'], 'Sentiment': sentiment})
        print("this worked")

        # Create DataFrame
        if not sentiment_data:
            dates = pd.date_range(start=start, end=end, freq='D').strftime('%Y-%m-%d').tolist()
            sentiment_df = pd.DataFrame({'Date': dates, 'Sentiment': [0.0] * len(dates)})
            return sentiment_df, 0.0, None

        print("the function is called 8.")
        sentiment_df = pd.DataFrame(sentiment_data)
        avg_sentiment = sentiment_df['Sentiment'].mean()

        # Fill missing dates with 0.0
        full_dates = pd.date_range(start=start, end=end, freq='D').strftime('%Y-%m-%d')
        full_df = pd.DataFrame({'Date': full_dates})
        sentiment_df = full_df.merge(sentiment_df, on='Date', how='left').fillna({'Sentiment': 0.0})

        print("The function works.")
        return sentiment_df, float(avg_sentiment), None

    except Exception as e:
        return None, None, f"Error: {str(e)}"