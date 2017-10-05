def argmax(values):
    """
    Given a list of values, return the index of the greatest value.
    """
    result, _ = max(enumerate(values), key=lambda pair: pair[1])
    return result

def transpose(sequences):
    """
    Given a sequence of sequences of equal length, transpose them, producing a
    sequence of the items in parallel (as tuples). Useful for batching.
    """
    return zip(*sequences)
