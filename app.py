from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import praw as praw
import pandas as pd
import csv
import datetime
import nltk
import matplotlib.pyplot as plt
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob



app = Flask(__name__,template_folder='E:/Sentiment Analyzer/',static_folder='static')
analyzer = SentimentIntensityAnalyzer()

import nltk
nltk.download('vader_lexicon')

reddit = praw.Reddit(
    client_id="mG0s_3OPp0QR85VVULmUDw",
    client_secret="TeNFxx-MeKinFbmTkfuh6sO_NpE5JA",
    password="Harshnema1234",
    user_agent="sentiment",
    username="nema_harsh" 
)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login.html')
def login():
    return render_template('login.html')



@app.route('/text.html')
def text():
    return render_template('text.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/service.html')
def service():
    return render_template('service.html')



@app.route('/audio.html')
def audio():
    return render_template('audio.html')


@app.route('/youtube.html')
def youtube():
    return render_template('youtube.html')


def check_word(word,data):
    contains=data['title'].str.contains(word,case=False)
    return contains

def get_sentiment(title):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(title)['compound']


@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.form.get('user-input')
    sentiment_score = perform_sentiment_analysis(text)
    return jsonify({'sentiment_score': sentiment_score})

def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']



@app.route('/custom_analyze', methods=['POST'])
def custom_analyze():
    try:
        data = request.get_json()
        url = data['url']

        # Define get_transcript function
        def get_transcript(url):
            video_id = url.split("v=")[1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript

        # Define analyze_sentiment1 function
        def analyze_sentiment1(transcript):
            text = ' '.join([entry['text'] for entry in transcript])
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            return sentiment_score

        transcript = get_transcript(url)
        if transcript:
            sentiment_score = analyze_sentiment1(transcript)
            return jsonify({'sentiment_score': sentiment_score})
        else:
            return jsonify({'error': 'Transcript not available for this video.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results.html', methods=['POST'])
def results():
    a = request.form['keyword']
    subreddit = reddit.subreddit(a)

    new_data_list = []
    for post in subreddit.hot(limit=90):
        new_data_list.append({
            "title": post.title,
            "author": post.author,
            "link": post.shortlink,
            "comment_ID": post.id,
            "time": datetime.datetime.fromtimestamp(post.created_utc)
        })

    field_names = ["title", "author", "link", "comment_ID", "time"]
    with open('product_sales.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(new_data_list)
    
    with open('product_sales.csv', 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data = list(csvreader)
    
    return render_template('results.html', keyword=a,data =data)



if __name__ == '__main__':
    app.run(debug=True, port= 5000)
