"""
Text processing utilities for conversation analysis.
Contains all text cleaning and conversation preprocessing functions.
"""

import re
import string
import pandas as pd


def clean_text_string(text, remove_punctuation=True):
    """
    Clean a single text string by removing transcription noise and optionally punctuation.
    
    Parameters:
    -----------
    text : str
        The input text to clean
    remove_punctuation : bool
        Whether to remove punctuation
        
    Returns:
    --------
    str
        The cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove text in parentheses, square brackets, and angle brackets
    text = re.sub(r'\([^)]*\)|\[[^\]]*\]|<[^>]*>', '', text)
    
    # Remove punctuation if requested
    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_text(df):
    """
    Clean conversation text by removing transcription noise and punctuation.
    """
    df_copy = df.copy()
    
    # Apply text cleaning to each text entry
    df_copy['text'] = df_copy['text'].apply(
        lambda x: clean_text_string(x, remove_punctuation=True) if isinstance(x, str) else x
    )
    
    # Remove rows with empty texts after cleaning
    df_copy = df_copy[df_copy['text'].astype(bool)]
    
    return df_copy


def utterances_to_turns(df):
    """
    Convert individual utterances to speaker turns by merging consecutive utterances 
    from the same speaker.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with columns: speaker, text, and other metadata
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with merged speaker turns
    """
    if df.empty:
        return df
    
    turns = []
    current_turn = None
    
    for _, row in df.iterrows():
        if current_turn is None:
            # Start first turn
            current_turn = row.to_dict()
        elif row['speaker'] == current_turn['speaker']:
            # Same speaker, merge text
            current_turn['text'] += ' ' + str(row['text'])
        else:
            # Different speaker, save current turn and start new one
            turns.append(current_turn)
            current_turn = row.to_dict()
    
    # Don't forget the last turn
    if current_turn is not None:
        turns.append(current_turn)
    
    return pd.DataFrame(turns) 