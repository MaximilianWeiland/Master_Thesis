import pandas as pd
import re
from typing import List, Tuple, Any

def create_regex_pattern(dictionary_df: pd.DataFrame) -> str:
    """
    Creates a combined regular expression pattern from a dictionary DataFrame.

    Each cell in the DataFrame is treated as a pattern. Patterns may contain
    asterisks (*) to indicate wildcards:
    - '*' as a standalone token matches any single word
    - '*word' matches any prefix ending in 'word'
    - 'word*' matches any suffix starting with 'word'

    Tokens within a pattern are matched with flexible whitespace, and all
    patterns are combined using a logical OR.

    Args:
        dictionary_df (pd.DataFrame): DataFrame containing dictionary patterns.

    Returns:
        str: A combined regular expression pattern matching all dictionary entries.
    """
    # get all patterns in the dictionary to look for
    patterns: List[str] = [word for col in dictionary_df.columns for word in dictionary_df[col] if pd.notna(word)]

    # initialize empty list to store the regex patterns
    regex_patterns: List[str] = []

    # loop through all patterns
    for pat in patterns:

        # split pattern into words by spaces
        tokens = pat.split()
        regex_parts: List[str] = []

        # loop through all tokens of the pattern
        for token in tokens:

            # match any word if asterisk is a separate token
            if token == '*':
                regex_parts.append(r'\w+')
            
            # match any prefix to the word
            elif token.startswith('*') and len(token) > 1:
                word = token[1:]
                regex_parts.append(rf'\w*{re.escape(word)}')
            
            # match any suffix to the word
            elif token.endswith('*') and len(token) > 1:
                word = token[:-1]
                regex_parts.append(rf'{re.escape(word)}\w*')

            # if no asterisk, take the word as is
            else:
                regex_parts.append(re.escape(token))

        # join tokens with \s+ to match spaces and append to the list of patterns
        regex_pattern = r'\b' + r'\s+'.join(regex_parts) + r'\b'
        regex_patterns.append(regex_pattern)

    # combine all regex patterns to one with or condition
    combined_regex: str = "|".join(regex_patterns)

    return combined_regex

def clean_text(text: str) -> str:
    """
    Cleans a text string from misencoded punctuation. Replaces misencoded characters
    by their actual UTF-8 encoding counterparts.

    Args:
        text (str): The text string to clean.

    Returns:
        str: The cleaned text string.
    """
    # return empty string if the text column is not a string
    if not isinstance(text, str):
        return ""

    # replace misencoded punctuation
    replacements = {
        "Â£": "£",
        "â€œ": "“",
        "â€": "”",
        "â€˜": "‘",
        "â€™": "’",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "â€": '"',
        "Ã©": "é",
    }
    for bad, good in replacements.items():
        text: str = text.replace(bad, good)

    # normalize whitespaces
    text: str = re.sub(r"\s+", " ", text)

    # remove leading and trailing whitespaces
    return text.strip()

def split_sentences(
        text: str,
        tokenizer: Any
) -> List[str]:
    """
    Splits sentences into indvidual words.

    Args:
        text (str): The text string to split up.
        tokenizer (Any): Tokenizer (from nltk library) used to split up words.
    
    Returns:
        List[str]: List containing all individual words
    """
    # return empty list if no text or text not a string
    if not isinstance(text, str) or not text.strip():
        return []
    # otherwise apply the tokenizer
    return tokenizer.tokenize(text)

def proportional_stratified_sample(
        df: pd.DataFrame,
        strata_cols: List[str],
        total_samples: int,
        random_state: int = 7
) -> pd.DataFrame:
    """
    Draws a proportional stratified sample from a DataFrame. The number of samples drawn from
    each stratum is proportional to the stratum's frequency in the original dataset.

    Args:
        df (pd.DataFrame): Input DataFrame to sample from.
        strata_cols (List[str]): Column names defining the strata.
        total_samples (int): Total number of samples to draw.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the stratified sample.
    """

    # get the size of all strata and their proportions
    stratum_counts: pd.Series = df.groupby(strata_cols).size()
    stratum_proportions: pd.Series = stratum_counts / stratum_counts.sum()
    
    # get the proportional count for each stratum
    stratum_sample_counts: pd.Series = (stratum_proportions * total_samples).round().astype(int)
    
    # empty list to store the samples for each stratum
    sampled_list: List[pd.DataFrame] = []

    # loop over all combinations
    for stratum_vals, n_samples in stratum_sample_counts.items():
        stratum_vals: Tuple
        n_samples: int

        # boolean mask (starts as scalar, becomes Series)
        mask: pd.Series | bool = True

        for col, val in zip(strata_cols, stratum_vals):
            mask &= df[col] == val

        # subset DataFrame for this stratum
        stratum_df: pd.DataFrame = df[mask]

        # cap sample size if stratum is small
        n_samples = min(n_samples, len(stratum_df))

        # sample rows and append to list
        sampled_df_part: pd.DataFrame = stratum_df.sample(
            n=n_samples,
            random_state=random_state
        )
        sampled_list.append(sampled_df_part)
    
    # concatenate all strata samples and return
    sampled_df: pd.DataFrame = pd.concat(sampled_list).reset_index(drop=True)

    return sampled_df