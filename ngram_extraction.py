"""
N-gram extraction from conversation dataframes.
Handles all n-gram extraction functionality including single text and corpus-level processing.
"""

import pandas as pd
import time
import sys
import re
import string
from collections import Counter
import os
import unicodedata
from .text_utils import clean_text_string


def extract_ngrams_from_text(text, n_gram_min=2, n_gram_max=3):
    """
    Extract n-grams from a single text within a specified range.
    
    Parameters:
    -----------
    text : str
        Input text
    n_gram_min : int
        Minimum n-gram length
    n_gram_max : int
        Maximum n-gram length
        
    Returns:
    --------
    list
        List of n-grams in the specified range
    """
    if not isinstance(text, str):
        return []
    
    input_list = text.split()
    ngrams_list = []
    
    for n in range(n_gram_min, n_gram_max + 1):
        if len(input_list) >= n:
            n_grams = [' '.join(input_list[i:i+n]) for i in range(len(input_list)-n+1)]
            ngrams_list.extend(n_grams)
    
    return ngrams_list


def clean_dataframe_text(df, text_column='text', remove_punctuation=True):
    """
    Clean text in a dataframe column.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    text_column : str
        The name of the column containing text to clean
    remove_punctuation : bool
        Whether to remove punctuation
        
    Returns:
    --------
    pandas DataFrame
        A copy of the input dataframe with cleaned text
    """
    df_copy = df.copy()
    
    # Apply text preprocessing to the specified column
    df_copy[text_column] = df_copy[text_column].apply(
        lambda x: clean_text_string(x, remove_punctuation=remove_punctuation)
    )
    
    # Remove rows with empty text after cleaning
    df_copy = df_copy[df_copy[text_column].astype(bool)]
    
    return df_copy






def extract_ngrams_from_corpus(df, text_column='text', n_gram_min=2, n_gram_max=3, 
                              preprocess=True, remove_punctuation=True, frequency_column='frequency'):
    """
    Extract n-grams from a dataframe's text column and count frequencies.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing text data
    text_column : str
        Name of the column containing text
    n_gram_min : int
        Minimum n-gram length to extract
    n_gram_max : int
        Maximum n-gram length to extract
    preprocess : bool
        Whether to preprocess the text before extracting n-grams
    remove_punctuation : bool
        Whether to remove punctuation during preprocessing
    frequency_column : str
        Name of the column to store n-gram frequencies
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with columns 'ngram', frequency_column, and 'ngram_length'
    """
    print(f"Started n-gram extraction from {len(df)} rows...")
    start_time = time.time()
    
    # Preprocess the dataframe if requested
    if preprocess:
        df = clean_dataframe_text(df, text_column=text_column, remove_punctuation=remove_punctuation)
    
    # Get all texts from the dataframe
    texts = df[text_column].astype(str).dropna().tolist()
    
    # Extract n-grams from each text
    all_ngrams = []
    last_progress_msg_len = 0
    for i, text in enumerate(texts):
        text_ngrams = extract_ngrams_from_text(text.lower(), n_gram_min, n_gram_max)
        all_ngrams.extend(text_ngrams)
        
        # Print progress every 1000 texts
        if (i + 1) % 1000 == 0:
            progress_msg = f"Processed {i+1}/{len(texts)} texts..."
            padding = " " * max(0, last_progress_msg_len - len(progress_msg))
            sys.stdout.write("\r" + progress_msg + padding)
            sys.stdout.flush()
            last_progress_msg_len = max(last_progress_msg_len, len(progress_msg))
    
    # Finish the progress line, if any
    if last_progress_msg_len > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Count n-gram frequencies
    ngram_counts = Counter(all_ngrams)
    
    # Convert to DataFrame
    result_df = pd.DataFrame([
        {'ngram': ngram, frequency_column: count, 'ngram_length': len(ngram.split())}
        for ngram, count in ngram_counts.items()
    ])
    
    # Sort by frequency in descending order
    result_df = result_df.sort_values(frequency_column, ascending=False).reset_index(drop=True)
    
    elapsed_time = time.time() - start_time
    print(f"Found {len(result_df)} unique n-grams")
    
    return result_df





def count_speakers_per_ngram(session_ngrams, session_turns, text_column='text'):
    """
    Count how many different speakers use each n-gram.
    
    Parameters:
    -----------
    session_ngrams : pandas DataFrame
        DataFrame with n-grams and their frequencies
    corpus_df : pandas DataFrame
        Original corpus dataframe with speaker and text columns
    text_column : str
        Name of the text column in corpus_df
        
    Returns:
    --------
    pandas DataFrame
        Original session_ngrams with added 'speaker_count' column
    """
    result_df = session_ngrams.copy()
    speaker_counts = {}
    
    # Get list of n-grams to check
    ngrams_to_check = result_df['ngram'].tolist()
    
    # Group by speaker to avoid counting multiple uses by same speaker
    for speaker, group in session_turns.groupby('speaker'):
        # Combine all text for this speaker and convert to lowercase
        speaker_text = ' '.join(group[text_column].astype(str).fillna('').str.lower())
        
        # Check each n-gram
        for ngram in ngrams_to_check:
            # Use proper word boundary checking to avoid partial matches
            # Add spaces around the speaker text and ngram to ensure proper word boundary matching
            padded_speaker_text = f" {speaker_text} "
            padded_ngram = f" {ngram} "
            if padded_ngram in padded_speaker_text:
                speaker_counts[ngram] = speaker_counts.get(ngram, 0) + 1
    
    # Add speaker counts to result dataframe
    result_df['speaker_count'] = result_df['ngram'].map(lambda x: speaker_counts.get(x, 0))
    
    return result_df 


def extract_ngrams_from_csv_folder(
    folder_path: str,
    *,
    text_column: str = 'text',
    n_gram_min: int = 1,
    n_gram_max: int = 3,
    chunksize: int = 250_000,
    encoding_primary: str = 'utf-8',
    encoding_fallback: str = 'latin1',
    remove_punctuation: bool = True,
    frequency_column: str = 'frequency_in_reference',
):
    """
    Stream and aggregate n-gram counts from multiple CSV files in a folder
    without loading them fully into memory.

    - Iterates CSVs in lexicographic order.
    - Reads each file in pandas chunks, normalizes Unicode (NFKC), cleans text
      using clean_text_string, and updates unigram & n-gram counts.

    Returns
    -------
    (ngrams_df, unigram_counts_series, total_unigrams)
        ngrams_df has columns: 'ngram', frequency_column, 'ngram_length'
    """
    ngram_counts: Counter[str] = Counter()
    unigram_counts: Counter[str] = Counter()

    csv_files = [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path))
        if fname.lower().endswith('.csv')
    ]

    total_rows_seen = 0
    for csv_path in csv_files:
        try:
            reader = pd.read_csv(
                csv_path,
                usecols=[text_column],
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines='skip',
                encoding=encoding_primary,
            )
        except UnicodeDecodeError:
            reader = pd.read_csv(
                csv_path,
                usecols=[text_column],
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines='skip',
                encoding=encoding_fallback,
            )

        for chunk in reader:
            texts = chunk[text_column].astype(str)
            for text in texts:
                # Normalize and clean
                normalized = unicodedata.normalize('NFKC', text)
                cleaned = clean_text_string(normalized, remove_punctuation=remove_punctuation)
                if not cleaned:
                    continue
                tokens = cleaned.lower().split()
                if not tokens:
                    continue
                # Update unigram counts
                unigram_counts.update(tokens)
                # Update n-gram counts across requested sizes
                for n in range(n_gram_min, n_gram_max + 1):
                    if len(tokens) >= n:
                        ngrams_iter = (' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
                        ngram_counts.update(ngrams_iter)
            total_rows_seen += len(chunk)

    # Build compact results
    ngrams_df = pd.DataFrame(
        {
            'ngram': list(ngram_counts.keys()),
            frequency_column: list(ngram_counts.values()),
        }
    )
    if not ngrams_df.empty:
        ngrams_df['ngram_length'] = ngrams_df['ngram'].str.split().str.len()
        ngrams_df = (
            ngrams_df.sort_values(frequency_column, ascending=False)
            .reset_index(drop=True)
        )

    total_unigrams = int(sum(unigram_counts.values()))
    unigram_series = pd.Series(unigram_counts)
    return ngrams_df, unigram_series, total_unigrams


def extract_counts_for_target_ngrams_from_csv_folder(
    folder_path: str,
    target_ngrams,
    *,
    text_column: str = 'text',
    chunksize: int = 250_000,
    encoding_primary: str = 'utf-8',
    encoding_fallback: str = 'latin1',
    remove_punctuation: bool = True,
    frequency_column: str = 'frequency_in_reference',
):
    """
    Much faster streaming extractor that only counts a provided set of target
    n-grams (e.g., the n-grams present in your session_ngrams). This avoids
    building the full reference n-gram universe.

    Parameters
    ----------
    folder_path : str
        Directory containing CSV files.
    target_ngrams : Iterable[str]
        N-gram strings to count in the reference corpus.

    Returns
    -------
    pandas.DataFrame
        Columns: 'ngram', frequency_column, 'ngram_length' for the targets found.
    """
    # Group target ngrams by length for efficient matching
    length_to_targets = {}
    for ng in target_ngrams:
        s = str(ng).strip().lower()
        if not s:
            continue
        L = len(s.split())
        if L <= 0:
            continue
        bucket = length_to_targets.get(L)
        if bucket is None:
            bucket = set()
            length_to_targets[L] = bucket
        bucket.add(s)

    counts = Counter()

    csv_files = [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path))
        if fname.lower().endswith('.csv')
    ]

    for csv_path in csv_files:
        try:
            reader = pd.read_csv(
                csv_path,
                usecols=[text_column],
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines='skip',
                encoding=encoding_primary,
            )
        except UnicodeDecodeError:
            reader = pd.read_csv(
                csv_path,
                usecols=[text_column],
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines='skip',
                encoding=encoding_fallback,
            )

        for chunk in reader:
            texts = chunk[text_column].astype(str)
            for text in texts:
                normalized = unicodedata.normalize('NFKC', text)
                cleaned = clean_text_string(normalized, remove_punctuation=remove_punctuation)
                if not cleaned:
                    continue
                tokens = cleaned.lower().split()
                if not tokens:
                    continue

                # For each requested n length, slide a window and update counts if present
                for n, targets in length_to_targets.items():
                    if len(tokens) < n:
                        continue
                    for i in range(len(tokens) - n + 1):
                        cand = ' '.join(tokens[i:i+n])
                        if cand in targets:
                            counts[cand] += 1

    if not counts:
        return pd.DataFrame(columns=['ngram', frequency_column, 'ngram_length'])

    result = pd.DataFrame(
        {
            'ngram': list(counts.keys()),
            frequency_column: list(counts.values()),
        }
    )
    result['ngram_length'] = result['ngram'].str.split().str.len()
    result = result.sort_values(frequency_column, ascending=False).reset_index(drop=True)
    return result