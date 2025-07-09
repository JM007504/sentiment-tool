from flask import Flask, render_template, request
import pandas as pd
from textblob import TextBlob

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    avg_scores = {'polarity': 0.0, 'subjectivity': 0.0}
    results = []
    error = None
    filename = None

    if request.method == 'POST':
        # Case 1: Manual Text Entry
        if 'user_text' in request.form and request.form['user_text'].strip():
            text = request.form['user_text'].strip()
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
            sentiment_counts[sentiment.lower()] += 1
            avg_scores['polarity'] = round(polarity, 2)
            avg_scores['subjectivity'] = round(subjectivity, 2)

            results.append({
                'text': text,
                'sentiment': sentiment,
                'polarity': round(polarity, 2),
                'subjectivity': round(subjectivity, 2)
            })

            filename = None  # Clear filename on text input

        # Case 2: File Upload
        elif 'file' in request.files:
            file = request.files['file']
            if file:
                filename = file.filename
                extension = filename.lower()
                polarities = []
                subjectivities = []

                try:
                    # --- CSV Processing ---
                    if extension.endswith('.csv'):
                        df = pd.read_csv(file)

                        if df.shape[1] != 1:
                            error = "CSV must contain only one column with text."
                        elif not df.columns[0].isidentifier():
                            error = "CSV header must be valid (no special characters or spaces)."
                        else:
                            for text in df.iloc[:, 0]:
                                blob = TextBlob(str(text))
                                polarity = blob.sentiment.polarity
                                subjectivity = blob.sentiment.subjectivity
                                polarities.append(polarity)
                                subjectivities.append(subjectivity)

                                if polarity > 0:
                                    sentiment_counts['positive'] += 1
                                elif polarity < 0:
                                    sentiment_counts['negative'] += 1
                                else:
                                    sentiment_counts['neutral'] += 1

                                results.append({
                                    'text': text,
                                    'sentiment': 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral',
                                    'polarity': round(polarity, 2),
                                    'subjectivity': round(subjectivity, 2)
                                })

                    # --- TXT Processing ---
                    elif extension.endswith('.txt'):
                        lines = file.read().decode('utf-8').splitlines()
                        text_lines = [line.strip() for line in lines if line.strip()]

                        if not text_lines:
                            error = "TXT file must contain valid text lines."

                        else:
                            for line in text_lines:
                                blob = TextBlob(line)
                                polarity = blob.sentiment.polarity
                                subjectivity = blob.sentiment.subjectivity
                                polarities.append(polarity)
                                subjectivities.append(subjectivity)

                                if polarity > 0:
                                    sentiment_counts['positive'] += 1
                                elif polarity < 0:
                                    sentiment_counts['negative'] += 1
                                else:
                                    sentiment_counts['neutral'] += 1

                                results.append({
                                    'text': line,
                                    'sentiment': 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral',
                                    'polarity': round(polarity, 2),
                                    'subjectivity': round(subjectivity, 2)
                                })

                    else:
                        error = "Please upload a valid .csv or .txt file only."

                    # Only compute scores if data is valid
                    if not error and polarities:
                        avg_scores['polarity'] = round(sum(polarities) / len(polarities), 2)
                        avg_scores['subjectivity'] = round(sum(subjectivities) / len(subjectivities), 2)

                except Exception as e:
                    error = f"Error processing file: {str(e)}"

    return render_template('index.html',
                           sentiment_counts=sentiment_counts,
                           avg_scores=avg_scores,
                           results=results,
                           error=error,
                           filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
