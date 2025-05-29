import re
import emoji
from collections import Counter, OrderedDict, defaultdict
from urllib.parse import urlparse


def find_repeated_sequences(text, min_length=5, tokenize_by='word'):
    """
    Removes repeated sequences of tokens from a given text.
    Works for any language, handling both word-based and character-based tokenization.
    
    :param text: The input string
    :param min_length: Minimum length of sequence to consider
    :param tokenize_by: 'word' for word-based, 'char' for character-based
    :return: The text with repeated sequences removed
    """
    # Tokenize the text
    if tokenize_by == 'word':
        tokens = text.split()
    else:
        tokens = list(text)  # Character-based tokenization
    
    n = len(tokens)
    sequence_counts = defaultdict(int)
    
    # Use a sliding window approach to check for repeated sequences
    for length in range(min_length, n // 2 + 1):  # Consider different sequence lengths
        seen = {}
        for i in range(n - length + 1):
            seq = tuple(tokens[i:i + length])  # Store as tuple for immutability
            if seq in seen:
                sequence_counts[seq] += 1
            else:
                seen[seq] = i
    
    # Remove all repeated sequences from the original tokens
    i = 0
    result = []
    while i < len(tokens):
        found_repetition = False
        for length in range(n // 2, min_length - 1, -1):  # Start from longest sequences
            seq = tuple(tokens[i:i + length])
            if sequence_counts.get(seq, 0) > 0:
                found_repetition = True
                i += length  # Skip the repeated sequence
                break
        if not found_repetition:
            result.append(tokens[i])
            i += 1
    
    return " ".join(result) if tokenize_by == 'word' else "".join(result)


def clean_text(text, lang):

    # languages to tokenize by characters
    char_lang = ["kor", "tha"] # Korean, Thai, what else?
    if lang in char_lang:
        tok_by = "char"
    else:
        tok_by = "word"

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove excessive punctuation
    text = re.sub(r'([,.!?;<>*])\1+', r'\1', text)
    
    # Remove duplicates
    sentences = re.split(r'[.!?]\s', text)  # Split by common sentence delimiters
    sentences = [find_repeated_sequences(s.strip(), tokenize_by=tok_by) for s in sentences if s.strip()]
    #sentences = [s.strip() for s in sentences if s.strip()]
    unique_sentences = list(OrderedDict.fromkeys(sentences))  # Preserve order, remove duplicates
    
    return '. '.join(unique_sentences) + '.'


def extract_meaningful_tokens_from_weblinks(text):

	#E.g., "https://www.technocracy.news/blaylock-face-masks-pose-serious-risks-to-the-healthy/"
	# will be converted to "https://www.technocracy.news/ blaylock face masks pose serious risks to the healthy"

    url_pattern = re.compile(r'(https?://\S+)')
    urls = url_pattern.findall(text)
    
    for url in urls:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path_tokens = re.split('[-_/]', parsed_url.path)
        path_tokens = [token for token in path_tokens if token and not token.isdigit()]
        
        extracted_text = f"{domain} {' '.join(path_tokens)}"
        text = text.replace(url, f"{url.split('/')[0]}//{extracted_text}")
    
    return text


def convert_hashtags(text):

	# E.g., "#MasksDoNotWork" will be converted to "masks do not work"

    def format_hashtag(match):
        words = re.sub(r'([A-Z])', r' \1', match.group(1)).strip()
        return f'"{words.lower()}"'
    
    return re.sub(r'#(\w+)', format_hashtag, text)
