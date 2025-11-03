"""
Mutual Information analysis for n-grams in conversation corpora.
Focuses on statistical analysis without text processing utilities.
"""

import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from .ngram_extraction import extract_ngrams_from_text


def analyze_mutual_information(
    df,
    text_column='text',
    n_gram_max=4,
    min_frequency=2,
    min_length=2,
    mi_column: str = 'mutual_information',
):
    """
    Analyze mutual information for n-grams in a conversation corpus.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with conversation data
    text_column : str
        Name of column containing text
    n_gram_max : int
        Maximum n-gram length to analyze
    min_frequency : int
        Minimum frequency threshold for n-grams
    min_length : int
        Minimum n-gram length to include
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with n-grams, frequencies, and MI scores
    """
    # Get utterances
    texts = list(df[text_column].dropna().astype(str))
    
    # Extract n-grams respecting sentence boundaries
    all_ngrams = []
    for text in texts:
        ngrams = extract_ngrams_from_text(text.lower(), 1, n_gram_max)
        all_ngrams.extend(ngrams)
    
    # Count n-grams
    ngram_counts = Counter(all_ngrams)
    
    # Create and filter DataFrame
    df_ngrams = pd.DataFrame.from_dict(ngram_counts, orient='index').reset_index()
    df_ngrams = df_ngrams.rename(columns={'index': 'ngram', 0: 'frequency'})
    df_ngrams['ngram_length'] = df_ngrams['ngram'].str.split().str.len()

    df_filtered = df_ngrams[
        (df_ngrams['ngram_length'] >= min_length)
        & (df_ngrams['frequency'] >= min_frequency)
    ].sort_values('frequency', ascending=False)
    
    # Prepare corpus for MI calculation
    corpus = ' '.join(text.lower() for text in texts)
    word_counts = Counter(corpus.split())
    
    # Pre-compute n-gram counts for optimization
    ngrams_to_analyze = list(df_filtered['ngram'])
    ngram_counts_dict = {ngram: corpus.count(ngram) for ngram in ngrams_to_analyze}
    
    # Calculate MI with optimized counting
    total_words = sum(word_counts.values())
    
    def calculate_mi(ngram):
        words = ngram.split()
        word_probs_product = 1
        
        for word in words:
            if word not in word_counts:
                return float('-inf')
            word_probs_product *= word_counts[word]/total_words
        
        ngram_count = ngram_counts_dict[ngram]
        if ngram_count == 0 or word_probs_product == 0:
            return float('-inf')
            
        return math.log2((ngram_count/total_words)/word_probs_product)
    
    df_filtered[mi_column] = [calculate_mi(ngram) for ngram in ngrams_to_analyze]
    return df_filtered.sort_values(mi_column, ascending=False)


def mi_from_tables(phrase_text: str, phrase_count: float, unigram_counts: pd.Series, total_unigrams: float, k: float = 0.0) -> float:
    # k=0 for exact counts; set small k (e.g., 0.5) if you want smoothing
    words = phrase_text.split()
    denom = 1.0
    N = float(total_unigrams)
    if N <= 0:
        return float('-inf')
    for w in words:
        uw = float(unigram_counts.get(w, 0.0))
        if k > 0:
            uw += k
        if uw <= 0:
            return float('-inf')
        denom *= (uw / (N + k * len(unigram_counts)))
    pc = float(phrase_count)
    if k > 0:
        pc += k
        N_eff = N + k  # minimal smoothing for phrase
    else:
        N_eff = N
    if pc <= 0 or denom <= 0:
        return float('-inf')
    return float(np.log2((pc / N_eff) / denom))


def npmi_from_tables(phrase_text: str, phrase_count: float, unigram_counts: pd.Series, total_unigrams: float, k: float = 0.0) -> float:
    """
    Compute normalized pointwise mutual information (NPMI) for a phrase using
    counts provided in tables.

    NPMI(phrase) = PMI(phrase) / (-log P(phrase))
                 = [log ( P(phrase) / Π P(w_i) )] / [-log P(phrase)]

    - Returns value in [-1, 1]. If probabilities are zero or undefined, returns -inf.
    - Optional Laplace-style smoothing via k.

    Parameters
    ----------
    phrase_text : str
        The n-gram text (space-separated tokens)
    phrase_count : float
        Observed count of the phrase in the corpus (unsmoothed)
    unigram_counts : pd.Series
        Series mapping token -> count
    total_unigrams : float
        Total number of tokens in the corpus
    k : float, optional
        Additive smoothing constant. If > 0, applies to unigrams and phrase.
    """
    # Compute PMI using the same smoothing scheme as mi_from_tables
    pmi = mi_from_tables(phrase_text, phrase_count, unigram_counts, total_unigrams, k=k)
    if not np.isfinite(pmi):
        return float('-inf')

    # Compute P(phrase)
    N = float(total_unigrams)
    pc = float(phrase_count)
    if k > 0:
        pc += k
        N_eff = N + k
    else:
        N_eff = N
    if N_eff <= 0 or pc <= 0:
        return float('-inf')
    p_phrase = pc / N_eff

    # Guard for numerical safety: -log(p) denominator
    if p_phrase <= 0.0:
        return float('-inf')
    denom = -math.log(p_phrase)
    if denom <= 0.0:
        # This only happens if p_phrase >= 1.0 due to extreme smoothing; treat as undefined
        return float('-inf')

    # Convert PMI from log2 to natural log scale to match denominator, or scale denom
    # pmi (log2) / [-log_e P]
    # Either convert pmi to natural logs: pmi_ln = pmi * ln(2)
    pmi_ln = float(pmi) * math.log(2.0)
    return pmi_ln / denom


def add_npmi_column(df: pd.DataFrame,
                    *,
                    phrase_col: str = 'ngram',
                    count_col: str = 'frequency',
                    unigram_counts: pd.Series,
                    total_unigrams: float,
                    out_col: str = 'npmi',
                    k: float = 0.0) -> pd.DataFrame:
    """
    Append an NPMI column to a DataFrame of phrases and counts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain phrase_col and count_col.
    unigram_counts : pd.Series
        Token -> count mapping for the corpus used to compute marginals.
    total_unigrams : float
        Total tokens in the corpus.
    out_col : str
        Name of the output column for NPMI values.
    k : float
        Optional smoothing added to counts.
    """
    if df is None or len(df) == 0:
        raise ValueError('DataFrame is empty')
    if phrase_col not in df.columns or count_col not in df.columns:
        raise ValueError(f"DataFrame must include columns {phrase_col!r} and {count_col!r}")

    result = df.copy()
    result[out_col] = [
        npmi_from_tables(str(p), float(c), unigram_counts, float(total_unigrams), k=k)
        for p, c in zip(result[phrase_col], result[count_col])
    ]
    return result


def analyze_npmi(
    df,
    text_column: str = 'text',
    n_gram_max: int = 4,
    min_frequency: int = 2,
    min_length: int = 2,
    out_col: str = 'npmi',
    smoothing_k: float = 0.0,
):
    """
    Analyze NPMI for n-grams in a conversation corpus, mirroring analyze_mutual_information.

    Returns a DataFrame with n-grams, frequencies, ngram_length, and an NPMI column.
    """
    # Get utterances
    texts = list(df[text_column].dropna().astype(str))

    # Extract n-grams (including unigrams for marginals)
    all_ngrams = []
    for text in texts:
        # Use local extraction to avoid circular import
        tokens = str(text).lower().split()
        for n in range(1, n_gram_max + 1):
            if len(tokens) >= n:
                all_ngrams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    # Count n-grams and unigrams
    ngram_counts = Counter(all_ngrams)
    unigram_counts = Counter([w for text in texts for w in str(text).lower().split()])

    # Build base DataFrame and filter
    df_ngrams = pd.DataFrame.from_dict(ngram_counts, orient='index').reset_index()
    df_ngrams = df_ngrams.rename(columns={'index': 'ngram', 0: 'frequency'})
    df_ngrams['ngram_length'] = df_ngrams['ngram'].str.split().str.len()
    df_filtered = df_ngrams[
        (df_ngrams['ngram_length'] >= min_length)
        & (df_ngrams['frequency'] >= min_frequency)
    ].sort_values('frequency', ascending=False).reset_index(drop=True)

    total_words = float(sum(unigram_counts.values()))
    # Compute NPMI per n-gram
    df_filtered[out_col] = [
        npmi_from_tables(ng, float(ct), pd.Series(unigram_counts), total_words, k=smoothing_k)
        for ng, ct in zip(df_filtered['ngram'], df_filtered['frequency'])
    ]
    return df_filtered.sort_values(out_col, ascending=False)