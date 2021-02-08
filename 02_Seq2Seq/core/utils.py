import re

def preprocess(string, vocab):
    string = clean_str(string)
    string = tokenize(string)
    string = [vocab(vocab.SOS_TOKEN)] + string + [vocab(vocab.EOS_TOKEN)]
    return string

def clean_str(string):
    """
    clean string from the input sentence to normalize it
    Args:
        string(str)
    Returns:
        (str)
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\|", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()

def tokenize(string):
    """
    Divide string to token
    Args:
        string(str)
    Returns:
        [str]
    """
    return string.split(' ')