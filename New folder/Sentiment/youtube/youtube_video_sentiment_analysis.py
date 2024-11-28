import os
from flask import Flask, request, jsonify, render_template
from googleapiclient.discovery import build
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__) 

# YouTube API Configuration
API_KEY = 'AIzaSyBk7vKl0otBEmYSsSBtlyB6S-l9ZoqQxNo'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Fetch YouTube Comments
def fetch_youtube_comments(video_id):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    comments_data = []
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText'
        )
        while request:
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                author_name = item['snippet']['topLevelComment']['snippet'].get('authorDisplayName', 'Anonymous')
                channel_id = item['snippet']['topLevelComment']['snippet'].get('authorChannelId', {}).get('value', 'Unknown')
                comments_data.append({'comment': comment, 'author': author_name, 'channel_id': channel_id})
            request = youtube.commentThreads().list_next(request, response)
    except Exception as e:
        print(f"Error: {e}")
    return comments_data

# Analyze Sentiments
def analyze_sentiments(comments_data):
    results = []
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    for data in comments_data:
        try:
            sentiment = sentiment_analyzer(data['comment'])[0]
            sentiment_label = sentiment['label']
            confidence = sentiment['score']

            if sentiment_label in ["1 star", "2 stars"]:
                overall_sentiment = "NEGATIVE"
                sentiment_counts['NEGATIVE'] += 1
            elif sentiment_label == "3 stars":
                overall_sentiment = "NEUTRAL"
                sentiment_counts['NEUTRAL'] += 1
            elif sentiment_label in ["4 stars", "5 stars"]:
                overall_sentiment = "POSITIVE"
                sentiment_counts['POSITIVE'] += 1
            else:
                overall_sentiment = "UNKNOWN"
            
            results.append({
                'comment': data['comment'],
                'author': data['author'],
                'channel_id': data['channel_id'],
                'sentiment': overall_sentiment,
                'confidence': round(confidence, 2)
            })
        except Exception as e:
            print(f"Error analyzing comment: {data['comment']} | {e}")
            results.append({
                'comment': data['comment'],
                'author': data['author'],
                'channel_id': data['channel_id'],
                'sentiment': 'Error',
                'confidence': 0
            })

    # Determine overall sentiment based on counts
    total_comments = sum(sentiment_counts.values())
    if total_comments == 0:
        overall_sentiment = "UNKNOWN"
    else:
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return results, sentiment_counts, overall_sentiment

# Generate Graph
def generate_sentiment_graph(sentiment_counts):
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['green', 'blue', 'red']
    explode = (0.1, 0, 0)  # Explode the first slice for better visualization

    plt.figure(figsize=(6, 4))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title("Sentiment Distribution")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_url = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close()

    return f"data:image/png;base64,{graph_url}"

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Analyze Route
@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form['video_url']
    if "v=" not in video_url:
        return jsonify({'error': 'Invalid YouTube URL'})
    video_id = video_url.split("v=")[-1]
    comments_data = fetch_youtube_comments(video_id)
    if not comments_data:
        return jsonify({'error': 'No comments found or invalid video ID'})
    
    sentiment_results, sentiment_counts, overall_sentiment = analyze_sentiments(comments_data)
    graph_url = generate_sentiment_graph(sentiment_counts)
    
    return jsonify({
        'results': sentiment_results,
        'positive_count': sentiment_counts['POSITIVE'],
        'negative_count': sentiment_counts['NEGATIVE'],
        'neutral_count': sentiment_counts['NEUTRAL'],
        'overall_sentiment': overall_sentiment,
        'graph_url': graph_url
    })

if __name__ == "__main__":
    app.run(debug=True)