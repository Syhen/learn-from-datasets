"""
@created by: heyao
@created at: 2021-12-08 20:49:50
"""
import numpy as np
from tqdm.auto import tqdm


def load_word_embedding(embedding_file, word_indexes, vector_size=50):
    embeddings = np.zeros((len(word_indexes) + 1, vector_size))
    # embeddings = np.random.random((len(word_indexes) + 1, vector_size))
    with open(embedding_file, "r", encoding="utf8") as f:
        # line = f.readline().strip()
        # _, n = line.split(" ")
        n = 500000
        tqdm_obj = tqdm(total=int(n))
        while 1:
            tqdm_obj.update(1)
            line = f.readline().strip()
            if not line:
                break
            word, vectors = line.split(" ", 1)
            if word not in word_indexes:
                continue
            embeddings[word_indexes[word]] = np.array([float(i) for i in vectors.split(" ")])
    return embeddings


if __name__ == '__main__':
    embeddings = load_word_embedding("/Users/heyao/Downloads/glove.6B.50d.txt", {"people": 0, "how": 1})
    print(embeddings.shape)
