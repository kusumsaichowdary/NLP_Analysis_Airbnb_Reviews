#Phase1_SentimentalCode.py
import pandas as pd
import nltk
import re
import emoji
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

# Step 1: Load Data
csv_file_path = 'NYC_2021_airbnb_reviews_data1.csv'
print("Loading data...")
data = pd.read_csv(csv_file_path)

# Step 2: Date Parsing
print("Parsing dates...")
data['review_posted_date'] = pd.to_datetime(data['review_posted_date'], format='%B %Y', errors='coerce')
data = data.dropna(subset=['review_posted_date'])  # Drop rows with invalid dates

# Step 3: Handle Missing Values
print("Handling missing values...")
if 'review' in data.columns:
    num_missing_reviews = data['review'].isna().sum()
    if num_missing_reviews > 0:
        print(f"Number of missing reviews: {num_missing_reviews}")
        data.dropna(subset=['review'], inplace=True)
    data.reset_index(drop=True, inplace=True)
else:
    raise KeyError("The dataset does not contain a 'review' column.")

# Step 4: Detect Language
print("Detecting language...")
from langdetect import detect
tqdm.pandas(desc="Detecting language")

def detect_language(text):
    try:
        return detect(text.strip()) if len(text.strip()) > 3 else 'unknown'
    except Exception:
        return 'unknown'

data['language'] = data['review'].progress_apply(detect_language)
data = data[data['language'] == 'en']
data.reset_index(drop=True, inplace=True)

# Step 5: Text Preprocessing
print("Expanding contractions and cleaning text...")
import contractions

def expand_contractions(text):
    try:
        return contractions.fix(text)
    except Exception:
        return text

def clean_text(text):
    try:
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = emoji.demojize(text)          # Convert emojis to text descriptions
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
        # Keep colons and underscores to preserve emoji descriptions
        text = re.sub(r'[^a-zA-Z0-9\s_:]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    except Exception:
        return text

data['review_expanded'] = data['review'].apply(expand_contractions)
data['review_cleaned_text'] = data['review_expanded'].apply(clean_text)


# Preprocess further for tokenization
print("Further preprocessing for tokenization...")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {
    'not', 'no', 'nor', 'never',   # Negations
    'very', 'extremely', 'really', # Intensifiers
    'i', 'we', 'my', 'you', 'your', 'our', 'us',   # Pronouns (lowercased)
}

def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['cleaned_review'] = data['review_cleaned_text'].apply(preprocess_text)

# Step 6: Initial Sentiment Analysis with Hugging Face Dataset
print("Performing initial sentiment analysis efficiently...")
sentiment_pipeline = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=0  # Set to -1 if no GPU available
)

# Create a Hugging Face Dataset from the data
review_dataset = Dataset.from_pandas(data[['cleaned_review']])

# Apply the sentiment pipeline directly to the dataset
def analyze_sentiment(batch):
    # Truncate reviews to a maximum of 512 characters
    truncated_reviews = [review[:512] for review in batch['cleaned_review']]
    batch['sentiment'] = sentiment_pipeline(truncated_reviews)
    return batch

# Map the function to the dataset
review_dataset = review_dataset.map(analyze_sentiment, batched=True)

# Extract sentiment results
data['initial_sentiment'] = [result['label'] for result in review_dataset['sentiment']]
data['sentiment_label'] = data['initial_sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})

# Step 7: Train-Test Split with Stratification
print("Splitting data into training and validation sets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['cleaned_review'], data['sentiment_label'], test_size=0.2, random_state=42, stratify=data['sentiment_label']
)

# Step 8: Prepare Dataset for Hugging Face
print("Preparing datasets...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['cleaned_review'],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_data = Dataset.from_pandas(pd.DataFrame({'cleaned_review': train_texts, 'label': train_labels}))
val_data = Dataset.from_pandas(pd.DataFrame({'cleaned_review': val_texts, 'label': val_labels}))

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 9: Define Model and Trainer
print("Initializing BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    eval_strategy='epoch',  # Updated to the correct argument
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

# Step 10: Train and Evaluate
print("Training the model...")
trainer.train()
print("Evaluating the model...")
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save Processed Data
print("Saving processed data...")
data.to_csv("processed_reviews_SentimentLabels.csv", index=False)
print("Process completed successfully!")
