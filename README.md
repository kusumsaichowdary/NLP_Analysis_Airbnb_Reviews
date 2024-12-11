# Comprehensive NLP Analysis of Airbnb Reviews for Improved Decision-Making

## Overview
This project aims to transform unstructured Airbnb review data into actionable insights using advanced Natural Language Processing (NLP) techniques. By performing sentiment analysis, aspect-based sentiment classification, and automated summarization, we enable hosts, guests, and stakeholders to quickly derive value from large volumes of textual feedback. The insights gained can inform business improvements, guide travelers’ booking decisions, and enhance user experience on the platform.

## Key Objectives
- **Sentiment Analysis:** Classify Airbnb reviews into positive or negative categories.
- **Aspect-Based Sentiment Analysis (ABSA):** Identify key aspects (e.g., location, cleanliness, host interaction) from the reviews and determine the sentiment polarity for each aspect.
- **Summarization:** Generate concise, coherent summaries of multiple reviews for a given listing, allowing users to grasp the overall sentiment and key themes quickly.

---

## Data Description
- **Input Dataset:** `NYC_2021_airbnb_reviews_data1.csv`
- **Columns:**
  - `listing_id`: Unique identifier for each Airbnb listing.
  - `url`: Link to the Airbnb listing.
  - `review_posted_date`: Date (month, year) when the review was posted.
  - `review`: Textual content of the guest review.

---

## Prerequisites and Setup

### Python Version
- **Recommended:** Python 3.8 or higher  
  Using a recent version ensures compatibility with the latest libraries.

### Environment
- Use a virtual environment (e.g., `venv` or `conda`) to isolate dependencies and avoid conflicts.

### GPU/CPU
- **Recommended:** GPU acceleration for faster model training and inference.  
  Scripts will run on CPU if a GPU is unavailable, though operations may be slower.

### Python Libraries and Dependencies
The following Python libraries are required:
- `pandas`: Data manipulation and CSV handling.
- `numpy`: Numerical computations.
- `nltk`: Tokenization, lemmatization, and POS tagging.
- `langdetect`: Language detection and filtering non-English reviews.
- `contractions`: Expanding contractions in text.
- `emoji`: Handling and converting emojis to text.
- `tqdm`: Progress bars for processing large datasets.
- `transformers`: Pre-trained models (e.g., BERT, T5) for sentiment analysis and summarization.
- `datasets`: Managing datasets compatible with `transformers`.
- `scikit-learn`: Metrics and clustering (e.g., KMeans).
- `gensim`: Training Word2Vec models for aspect extraction.
- `matplotlib`: Creating visualizations and plots.
- `seaborn`: Advanced statistical plots.
- `wordcloud`: Generating word clouds.
- `re`: Regular expressions for text cleaning.

#### Sample Installation Command:
pip install pandas numpy nltk langdetect contractions emoji tqdm transformers datasets scikit-learn gensim matplotlib seaborn wordcloud

# NLTK Data

**Note:** The code uses several NLTK resources to process text. Ensure these resources are downloaded before running the scripts. The following NLTK packages are used:  
- **punkt**: For tokenization.  
- **wordnet**: For lemmatization.  
- **stopwords**: For removing common English stopwords.  
- **averaged_perceptron_tagger**: For POS tagging (to identify nouns for aspect extraction).

# Scripts and Usage

## 1. Sentiment Analysis Pipeline (`Phase1_SentimentalCode.py`)

**Purpose:**  
- Load and preprocess Airbnb reviews.  
- Detect language and filter only English reviews.  
- Expand contractions, clean text, tokenize, and lemmatize.  
- Generate initial sentiment labels using a pre-trained DistilBERT sentiment model.  
- Fine-tune a BERT model for sentiment classification.  
- Save the processed dataset with sentiment labels.

**Run:**  
```bash
python Phase1_SentimentalCode.py
