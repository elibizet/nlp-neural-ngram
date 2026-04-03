"""
The word vector viewer is used to inspect embeddings 
by finding similar words using cosine similarity.
e.g. python3 wvv.py embeddings.vec Friday
"""

import sys
import math


def load_vectors(path):
    words = []
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError("First line must contain: vocab_size dimension")

        vocab_size, dim = map(int, header)

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            word = parts[0]
            vector = [float(x) for x in parts[1:]]

            if len(vector) != dim:
                raise ValueError(f"Vector for {word} has wrong dimension")

            words.append(word)
            vectors.append(vector)

    if len(words) != vocab_size:
        print(f"Warning: header says {vocab_size} words, but found {len(words)}")

    return words, vectors


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a, b):
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def most_similar(words, vectors, query_word, top_n=10):
    word_to_index = {word: i for i, word in enumerate(words)}

    if query_word not in word_to_index:
        print(f"Word not found: {query_word}")
        return

    query_idx = word_to_index[query_word]
    query_vec = vectors[query_idx]

    scores = []
    for i, word in enumerate(words):
        sim = cosine_similarity(query_vec, vectors[i])
        scores.append((sim, word))

    scores.sort(key=lambda x: x[0], reverse=True)

    for sim, word in scores[:top_n]:
        print(f"{sim:.3f} {word}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 wvv.py myvectors.vec word [top_n]")
        return

    path = sys.argv[1]
    query_word = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    words, vectors = load_vectors(path)
    most_similar(words, vectors, query_word, top_n)


if __name__ == "__main__":
    main()