from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score
if __name__ == "__main__":
    sample_texts = [
        "I love this product! It's amazing.",
        "I hate waiting in long lines.",
        "I'm not sure how I feel about this.",
    ]
    
    for text in sample_texts:
        sentiment, score = analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}, Confidence Score: {score:.4f}")
        print("="*50)
