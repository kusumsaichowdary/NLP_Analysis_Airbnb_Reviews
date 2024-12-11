README
Project Title: Comprehensive NLP Analysis of Airbnb Reviews for Improved Decision-Making
Overview
This project aims to transform unstructured Airbnb review data into actionable insights using advanced Natural Language Processing (NLP) techniques. By performing sentiment analysis, aspect-based sentiment classification, and automated summarization, we enable hosts, guests, and stakeholders to quickly derive value from large volumes of textual feedback. The insights gained can inform business improvements, guide travelers’ booking decisions, and enhance user experience on the platform.
Key Objectives:
1. Sentiment Analysis: Classify Airbnb reviews into positive or negative categories.
2. Aspect-Based Sentiment Analysis (ABSA): Identify key aspects (e.g., location, cleanliness, host interaction) from the reviews and determine the sentiment polarity for each aspect.
3. Summarization: Generate concise, coherent summaries of multiple reviews for a given listing, allowing users to grasp the overall sentiment and key themes quickly.
Data Description
* Input Dataset: NYC_2021_airbnb_reviews_data1.csv
   * Columns:
      * listing_id: Unique identifier for each Airbnb listing
      * url: Link to the Airbnb listing
      * review_posted_date: Date (month, year) when the review was posted
      * review: Textual content of the guest review
Prerequisites and Setup
Python Version
* Recommended: Python 3.8 or higher
Using a recent version of Python ensures compatibility with the latest libraries.
Environment
   * It is highly recommended to use a virtual environment (e.g., venv or conda) to keep project dependencies isolated and avoid conflicts with other projects.
GPU/CPU
   * GPU acceleration is highly recommended for faster model training and inference, but not strictly required. If you do not have a GPU, the scripts will run on CPU, though operations might be slower.
Python Libraries and Dependencies
Below is a list of all the Python libraries and dependencies used throughout the codebase:
   1. pandas: For data manipulation and reading CSV files.
   2. numpy: For numerical computations and array handling.
   3. nltk: For natural language processing tasks such as tokenization, lemmatization, and POS tagging.
   4. langdetect: For detecting the language of each review and filtering non-English content.
   5. contractions: To expand English language contractions in the review text.
   6. emoji: To handle and convert emojis to text descriptions.
   7. tqdm: For progress bars, especially when applying functions to large datasets.
   8. transformers (Hugging Face): For pre-trained models (BERT, T5) for sentiment analysis, classification, and summarization.
   9. datasets (Hugging Face): For creating and managing datasets compatible with the transformers library.
   10. scikit-learn (sklearn): For model evaluation metrics, and clustering (KMeans) in aspect-based sentiment analysis.
   11. gensim: For training Word2Vec models to generate word embeddings for aspect extraction.
   12. matplotlib: For plotting figures, charts, and visualizations.
   13. seaborn: For advanced plotting styles and statistical data visualization.
   14. wordcloud: For generating word cloud visualizations of common terms.
   15. re (Regular Expressions): For advanced text cleaning operations.
Sample installation command:
pip install pandas numpy nltk langdetect contractions emoji tqdm transformers datasets scikit-learn gensim matplotlib seaborn wordcloud
NLTK Data
The code uses several NLTK resources to process text. Ensure these resources are downloaded before running the scripts. The following NLTK packages are used:
   * punkt: For tokenization.
   * wordnet: For lemmatization.
   * stopwords: For removing common English stopwords.
   * averaged_perceptron_tagger: For POS tagging (to identify nouns for aspect extraction).
Scripts and Usage
1. Sentiment Analysis Pipeline (Phase1_SentimentalCode.py)
Purpose:
   * Load and preprocess Airbnb reviews.
   * Detect language and filter only English reviews.
   * Expand contractions, clean text, tokenize, and lemmatize.
   * Generate initial sentiment labels using a pre-trained DistilBERT sentiment model.
   * Fine-tune a BERT model for sentiment classification.
   * Save the processed dataset with sentiment labels.
Run:
python Phase1_SentimentalCode.py
Output:
   * processed_reviews_SentimentLabels.csv in data/ directory.
   * Console logs showing training and evaluation metrics.
2. Sentiment Visualization (Sentiment_plots.ipynb)
Purpose:
   * Visualize the distribution of sentiments.
   * Create bar charts for sentiment distribution, time-series line charts of yearly trends, and generate word clouds and bigram frequency charts for positive and negative reviews.
Run:
jupyter notebook Sentiment_plots.ipynb
Open the notebook in a browser and run the cells to produce the plots.
3. Aspect-Based Sentiment Analysis (ABSA_DataProcessing.py)
Purpose:
   * Load processed reviews with sentiment labels.
   * Tokenize and lemmatize reviews for Word2Vec model training.
   * Train Word2Vec to produce embeddings for vocabulary.
   * Extract nouns and cluster them using KMeans to identify aspects.
   * Map aspects to each review and create an aspect-level sentiment dataset.
Run:
python ABSA_DataProcessing.py
Output:
   * AspectbasedSentimentAnalysis.csv with listing_id, review_posted_date, cleaned_review, aspects, and sentiment_label.
4. Aspect-Based Visualization (ABSA_plots.ipynb)
Purpose:
   * Load AspectbasedSentimentAnalysis.csv
   * Create visualizations such as heatmaps for sentiment per aspect, radar charts for specific listings, and stacked area charts to show aspect sentiment evolution over time.
open as a notebook if provided in .ipynb format.
5. Summarization (FinalSummarization.py)
Purpose:
   * Prompt user for a listing_id. Example: 10452
   * Summarize all reviews associated with that listing using a T5 model.
   * Provides a concise narrative capturing host personality, accommodation quality, neighborhood vibe, safety, and transportation options.
Run:
python FinalSummarization.py
Output:
   * A final summary displayed in the console.
Interpreting the Results
   * Sentiment Analysis Results:
High accuracy and F1 scores indicate the model’s strong performance in classifying reviews. Positive reviews dominate, suggesting general guest satisfaction.
   * Aspect-Based Insights:
Clustering nouns into aspects reveals what specifically guests value (e.g., location, cleanliness) and what they dislike (e.g., cancellations).
The aspect-level sentiment dataset allows you to identify areas needing attention and improvement.
   * Visualizations:
The provided plots (bar charts, heatmaps, line graphs, radar charts, stacked area plots, word clouds) offer a comprehensive view of sentiment trends, key words, aspect distributions, and temporal changes.
   * Summarization:
Automatically generated summaries help users quickly understand the essence of thousands of reviews, focusing on critical attributes mentioned frequently by guests.
References
      * Dataset: NYC Airbnb Reviews 2021 (Kaggle)
      * BERT: Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
      * T5: Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
      * Word2Vec: Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases.
      * Hugging Face Transformers: https://huggingface.co/transformers/
      * NLTK: https://www.nltk.org/
Contact
For questions or feedback, please open an issue in this repository or contact the project maintainers directly.