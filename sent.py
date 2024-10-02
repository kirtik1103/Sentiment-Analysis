import os
import praw
import pandas as pd
import csv
import datetime
import matplotlib.pyplot as plt
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download NLTK's VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize Reddit API client using credentials from environment variables
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID', 'mG0s_3OPp0QR85VVULmUDw'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'TeNFxx-MeKinFbmTkfuh6sO_NpE5JA'),
    password=os.getenv('REDDIT_PASSWORD', 'Harshnema1234'),
    user_agent=os.getenv('REDDIT_USER_AGENT', 'sentiment'),
    username=os.getenv('REDDIT_USERNAME', 'nema_harsh')
)

# Prompt the user for a keyword to search
keyword = input("Enter the keyword to search: ")
subreddit = reddit.subreddit(keyword)

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Fetch top posts from the subreddit
top_posts = subreddit.top(limit=10)

# Analyze sentiment for each post and store results in a list
results = []
for post in top_posts:
    title = post.title
    sentiment = analyzer.polarity_scores(title)
    results.append({'title': title, 'sentiment': sentiment})

# Collect additional data
new_data_list = []
for post in subreddit.hot(limit=90):
    new_data_list.append({
        "title": post.title,
        "author": str(post.author),
        "link": post.shortlink,
        "comment_ID": post.id,
        "time": datetime.datetime.fromtimestamp(post.created_utc)
    })

# Write data to CSV
field_names = ["title", "author", "link", "comment_ID", "time"]
with open('product_sales.csv', 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(new_data_list)

# Read data from CSV
df = pd.read_csv(r'product_sales.csv')

# Function to check word in titles
def check_word(word, data):
    contains = data['title'].str.contains(word, case=False)
    return contains

# Check for keyword in titles
df[keyword] = check_word(keyword, df)

# Set the time column as index
df['time'] = pd.to_datetime(df['time'])
df = df.set_index("time")

# Resample and plot mentions over time
mean_keyword = df[keyword].resample('D').mean()
plt.plot(mean_keyword.index, mean_keyword, marker='o', color='r')
plt.title('Mentions over time')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.legend([keyword])
plt.xticks(rotation=45)
plt.show()

# Sentiment analysis on titles
sid = SentimentIntensityAnalyzer()
sentiment_score = df['title'].apply(sid.polarity_scores)
sentiment = sentiment_score.apply(lambda x: x['compound'])

# Print titles with high and low sentiment
if not df[sentiment > 0.6].empty:
    print(f"High sentiment post: {df[sentiment > 0.6]['title'].values[0]}")
else:
    print("No posts with high sentiment found.")

if not df[sentiment < 0.6].empty:
    print(f"Low sentiment post: {df[sentiment < 0.6]['title'].values[0]}")
else:
    print("No posts with low sentiment found.")

# Calculate and print average sentiment score for posts containing the keyword
avg_sentiment_score = sentiment[check_word(keyword, df)].mean()
print(f"Average sentiment score for '{keyword}': {avg_sentiment_score}")

# Determine overall sentiment
if avg_sentiment_score > 0.05:
    print("Overall sentiment is Positive")
elif avg_sentiment_score < -0.05:
    print("Overall sentiment is Negative")
else:
    print("Overall sentiment is Neutral")

# Resample and plot sentiment over time
sentiment_keyword = sentiment[check_word(keyword, df)].resample('D').mean()
plt.plot(sentiment_keyword.index, sentiment_keyword, color='blue', marker='o')
plt.xlabel('Date')
plt.ylabel('Sentiment')
plt.title(f"Sentiment of '{keyword}' over time")
plt.legend([keyword])
plt.xticks(rotation=45)
plt.show()

# Show sentiment classification for each post
df['sentiment'] = sentiment.apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
print(df[['title', 'sentiment']])
