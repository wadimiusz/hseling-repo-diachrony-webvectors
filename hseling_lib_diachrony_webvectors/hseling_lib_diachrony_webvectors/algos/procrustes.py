import gensim
import numpy as np
from utils import log
from typing import Optional
from tqdm.auto import tqdm


def smart_procrustes_align_gensim(base_embed: gensim.models.KeyedVectors,
                                  other_embed: gensim.models.KeyedVectors):
    """
    This code, taken from
    https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf and modified,
    uses procrustes analysis to make two word embeddings compatible.
    :param base_embed: first embedding
    :param other_embed: second embedding to be changed
    :return other_embed: changed embedding
    """
    base_embed.init_sims()
    other_embed.init_sims()

    shared_vocab = list(
        set(base_embed.vocab.keys()).intersection(other_embed.vocab.keys()))

    base_idx2word = {num: word for num, word in enumerate(base_embed.index2word)}
    other_idx2word = {num: word for num, word in enumerate(other_embed.index2word)}

    base_word2idx = {word: num for num, word in base_idx2word.items()}
    other_word2idx = {word: num for num, word in other_idx2word.items()}

    base_shared_indices = [base_word2idx[word] for word in
                           tqdm(shared_vocab)]  # remember to remove tqdm
    other_shared_indices = [other_word2idx[word] for word in
                            tqdm(shared_vocab)]  # remember to remove tqdm

    base_vecs = base_embed.syn0norm
    other_vecs = other_embed.syn0norm

    base_shared_vecs = base_vecs[base_shared_indices]
    other_shared_vecs = other_vecs[other_shared_indices]

    m = other_shared_vecs.T @ base_shared_vecs
    u, _, v = np.linalg.svd(m)
    ortho = u @ v

    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.syn0norm = other_embed.syn0 = other_embed.syn0norm.dot(ortho)

    return other_embed


class ProcrustesAligner(object):
    def __init__(self, w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors, already_aligned=True):
        self.w2v1 = w2v1
        if already_aligned:
            self.w2v2 = w2v2
        else:
            self.w2v2 = smart_procrustes_align_gensim(w2v1, w2v2)

    def __repr__(self):
        return "ProcrustesAligner"

    def get_score(self, word):
        vector1 = self.w2v1.wv[word]
        vector2 = self.w2v2.wv[word]
        score = np.dot(vector1, vector2)  # More straightforward computation
        return score

    def get_changes(self, top_n_changed_words: int, pos: Optional[str] = None):
        log('Doing procrustes')
        result = list()
        # their vocabs should be the same, so it doesn't matter over which to iterate:
        for word in set(self.w2v1.wv.vocab.keys()) & set(self.w2v2.wv.vocab.keys()):
            if pos is None or pos.lower() == "all" or word.endswith("_" + pos):
                score = self.get_score(word)
                result.append((word, score))

        result = sorted(result, key=lambda x: x[1])
        result = result[:top_n_changed_words]
        log('Done')
        return result


if __name__ == "__main__":
    pass
