# load all libraries necessary for the functions
import pandas as pd
import re

def create_regex_pattern(dictionary_df):
    # get all patterns in the dictionary to look for
    patterns = [word for col in dictionary_df.columns for word in dictionary_df[col] if pd.notna(word)]

    # initialize empty list to store the regex patterns
    regex_patterns = []

    # loop through all patterns
    for pat in patterns:

        # split pattern into words by spaces
        tokens = pat.split()
        regex_parts = []

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
    combined_regex = "|".join(regex_patterns)

    return combined_regex