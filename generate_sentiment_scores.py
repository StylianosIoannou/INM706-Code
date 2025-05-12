import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER sentiment lexicon
nltk.download('vader_lexicon')

# Load dataset
input_path = "archive/RedditNews.csv"
df = pd.read_csv(input_path)

# Initialize the VADER sentiment analyzer and calculate compound sentiment scores
SIA =SentimentIntensityAnalyzer()
df["sentiment"] = df["News"].apply(lambda text: SIA.polarity_scores(text)["compound"])

# Group the sentiment scores by date and rename columns for consistency and easier merging later
daily_sentiment = df.groupby("Date")["sentiment"].mean().reset_index()
daily_sentiment.columns = ["date", "sentiment"]

# Save the resulting daily sentiment scores to a new CSV file
output_path = "sentiment_scores.csv"
daily_sentiment.to_csv(output_path, index=False)

# Confirmation messege
print(f"Sentiment scores saved to: {output_path}")