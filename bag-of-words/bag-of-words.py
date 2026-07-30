import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    word_to_count = {}
    for token in tokens:
        if token in vocab:
            if token not in word_to_count:
                word_to_count[token] = 1
            else:
                word_to_count[token] += 1

    print(word_to_count)
    res = []
    for v in vocab:
        res.append(word_to_count.get(v, 0))
    return np.array(res).astype(int)