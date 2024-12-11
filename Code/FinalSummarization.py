import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

# Load data and prompt user for listing ID
csv_file_path = 'processed_reviews_SentimentLabels.csv'
data = pd.read_csv(csv_file_path)

listing_id = int(input("Please enter the listing_id to summarize reviews for: "))

# Initialize model and tokenizer
model_name = 't5-large'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def chunk_text(text, tokenizer, chunk_size=512):
    """
    Split a long text into smaller chunks based on tokenized length.

    Args:
        text (str): Input text to be chunked.
        tokenizer (T5Tokenizer): Tokenizer for encoding the text.
        chunk_size (int): Maximum number of tokens per chunk.

    Returns:
        list[torch.Tensor]: A list of tokenized chunks.
    """
    tokenized_text = tokenizer.encode(text, return_tensors="pt")[0]
    return [tokenized_text[i:i + chunk_size] for i in range(0, len(tokenized_text), chunk_size)]


def summarize_chunk(chunk, tokenizer, model, device, chunk_max_length=100, verbose=True):
    """
    Generate a summary for a single chunk of text.

    Args:
        chunk (torch.Tensor): Tokenized chunk of text.
        tokenizer (T5Tokenizer): Tokenizer for decoding and preparing input.
        model (T5ForConditionalGeneration): Pre-trained T5 model for summarization.
        device (torch.device): Device to execute the model on (CPU/GPU).
        chunk_max_length (int): Maximum length of the summary.
        verbose (bool): Whether to print detailed logs.

    Returns:
        str: Generated summary for the chunk.
    """
    chunk_text_decoded = tokenizer.decode(chunk, skip_special_tokens=True)
    input_text = "summarize: " + chunk_text_decoded
    tokenized_text = tokenizer.encode(input_text, return_tensors="pt").to(device)

    summary_ids = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        min_length=30,
        max_length=chunk_max_length,
        early_stopping=True
    )
    chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if verbose:
        print(f"[CHUNK INPUT]: {input_text[:300]}...")
        print(f"[CHUNK SUMMARY]: {chunk_summary}")

    return chunk_summary


def summarize_large_text(text, tokenizer, model, device, chunk_size=512, chunk_max_length=100,
                         final_min_length=100, final_max_length=200, verbose=True):
    """
    Summarize a large text by breaking it into chunks, summarizing each, and combining the results.

    Args:
        text (str): Input text to be summarized.
        tokenizer (T5Tokenizer): Tokenizer for encoding and decoding.
        model (T5ForConditionalGeneration): Pre-trained T5 model for summarization.
        device (torch.device): Device to execute the model on (CPU/GPU).
        chunk_size (int): Maximum token size for each chunk.
        chunk_max_length (int): Maximum length for each chunk's summary.
        final_min_length (int): Minimum length for the final summary.
        final_max_length (int): Maximum length for the final summary.
        verbose (bool): Whether to print detailed logs.

    Returns:
        str: Final comprehensive summary.
    """
    if verbose:
        print("\nSplitting text into chunks...")
    chunks = chunk_text(text, tokenizer, chunk_size)
    if verbose:
        print(f"Total chunks created: {len(chunks)}")

    if verbose:
        print("\nSummarizing individual chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        if verbose:
            print(f"\nSummarizing chunk {i}/{len(chunks)}")
        chunk_summary = summarize_chunk(chunk, tokenizer, model, device, chunk_max_length, verbose=verbose)
        chunk_summaries.append(chunk_summary)

    combined_summary_text = " ".join(chunk_summaries)

    final_input_text = (
            "summarize: Provide a detailed and coherent summary of guest reviews, "
            "highlighting themes about the host's personality, accommodation quality, neighborhood vibe, "
            "safety, and accessibility to transportation. " + combined_summary_text
    )

    tokenized_summary = tokenizer.encode(final_input_text, return_tensors="pt").to(device)
    if verbose:
        print("\nGenerating final summary...")

    summary_ids = model.generate(
        tokenized_summary,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        min_length=final_min_length,
        max_length=final_max_length,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_reviews_for_listing(listing_id, verbose=True):
    """
    Summarize all reviews associated with a given listing ID.

    Args:
        listing_id (int): ID of the listing to summarize reviews for.
        verbose (bool): Whether to print detailed progress and logs.

    Returns:
        str: Final summary of reviews for the listing ID.
    """
    if verbose:
        print(f"\nRetrieving reviews for listing_id {listing_id}...")
    reviews = data[data['listing_id'] == listing_id]['review']
    if reviews.empty:
        return f"No reviews found for listing_id {listing_id}"

    if verbose:
        print("\nSample reviews:")
        for idx, review in enumerate(reviews.head(3), start=1):
            print(f"[Sample Review {idx}]: {review}")

    combined_reviews = " ".join(reviews)

    if verbose:
        print("\nSummarizing all reviews...")
    return summarize_large_text(
        combined_reviews,
        tokenizer,
        model,
        device,
        chunk_size=512,
        chunk_max_length=100,
        final_min_length=100,
        final_max_length=200,
        verbose=verbose
    )


# Execute the summarization
final_summary = summarize_reviews_for_listing(listing_id, verbose=True)
print(f"\nFINAL SUMMARY for listing_id {listing_id}:\n{final_summary}")
