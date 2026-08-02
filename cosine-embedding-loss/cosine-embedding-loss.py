def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    dot_prod = sum(a*b for a, b in zip(x1, x2))
    norm_x1 = math.sqrt(sum(a**2 for a in x1))
    norm_x2 = math.sqrt(sum(a**2 for a in x2))
    cos_sim = dot_prod / (norm_x1 * norm_x2)
    if label == 1:
        return (1 - cos_sim)
    if label == -1:
        return max(0, cos_sim - margin)