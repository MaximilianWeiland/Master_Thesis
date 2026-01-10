import pandas as pd
import re
from typing import List, Dict, Tuple, Any

def create_regex_pattern(dictionary_df: pd.DataFrame) -> re.Pattern:
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
        re.Pattern: A combined regular expression pattern matching all dictionary entries.
    """
    # get all patterns in the dictionary to look for
    patterns: List[str] = [word for col in dictionary_df.columns for word in dictionary_df[col] if pd.notna(word)]

    # initialize empty list to store the regex patterns
    regex_patterns: List[str] = []

    # loop over all patterns
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

def __normalize_group_name(name: str) -> str:
    """
    Helper function that normalizes the name a social group so that it can be stored
    as a category within the regular expression pattern.

    Args:
        name (str): Name of the social group

    Returns:
        str: Normalized name of the social group by removing whitespaces and replacing with underscore
    """
    # get rid of leading and trailing whitespaces
    name: str = name.strip()

    # replace whitespaces with underscore
    name: str = re.sub(r'\W+', '_', name)

    # handle names consisting of digits
    if name[0].isdigit():
        name = f"cat_{name}"

    return name

def create_category_regex(dictionary_df: pd.DataFrame) -> re.Pattern:
    """
    Creates a combined regular expression pattern from a dictionary DataFrame.
    Stores additionally the group name of each dictionary entry.

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
        re.Pattern: A combined regular expression pattern matching all dictionary entries.
    """
    # empty list to store all individual patterns in
    category_patterns: List[str] = []

    # dictionary to map safe group name to original category name
    group_to_category: Dict[str, str] = {}

    # loop over all dictionary categories
    for category in dictionary_df.columns:

        # normalize group name and append to dictionary
        safe_name: str = __normalize_group_name(category)
        group_to_category[safe_name] = category
        # get all valid patterns
        patterns = dictionary_df[category].dropna()
        # empty list to store patterns for this category
        regex_patterns = []

        # loop over all patterns within the category
        for pat in patterns:

            # split into indvidual words/tokens
            tokens: str = pat.split()
            regex_parts: List[str] = []

            # loop over all tokens and append regular expression to list
            for token in tokens:
                if token == '*':
                    regex_parts.append(r'\w+')
                elif token.startswith('*') and len(token) > 1:
                    word = token[1:]
                    regex_parts.append(rf'\w*{re.escape(word)}')
                elif token.endswith('*') and len(token) > 1:
                    word = token[:-1]
                    regex_parts.append(rf'{re.escape(word)}\w*')
                else:
                    regex_parts.append(re.escape(token))

            # combine all regex parts for the single pattern and append
            regex_patterns.append(
                r'\b' + r'\s+'.join(regex_parts) + r'\b'
            )

        # append all category patterns to the general list
        if regex_patterns:
            category_patterns.append(
                f"(?P<{safe_name}>{'|'.join(regex_patterns)})"
            )

    # combine all patterns across categories to a single regex
    combined: str = "|".join(category_patterns)

    return re.compile(combined, flags=re.IGNORECASE), group_to_category

def match_dictionary(
        category_regex: str,
        group_lookup: Dict[str, str],
        text: str
) -> Tuple[bool, str]:
    """
    Applies regular expression to a text string.
    Returns if it found a match and if so the corresponding social group.

    Args:
        category_regex (str): Regular expression with social group search terms
        group_lookup (Dict[str, str]): Dictionary mapping safe social group names to the original ones
        text (str): The text string to search in

    Returns:
        Tuple[bool, str]:
            - Boolean indicating if there is a match
            - Actual social group name if there is a match, None otherwise
    """
    m = category_regex.search(text)
    if not m:
        return False, None
    return True, group_lookup[m.lastgroup]

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