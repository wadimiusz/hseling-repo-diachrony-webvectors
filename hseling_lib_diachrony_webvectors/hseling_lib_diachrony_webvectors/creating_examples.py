import pandas as pd
import numpy as np
from utils import log, format_time, intersection_align_gensim
from algos import smart_procrustes_align_gensim
import time
from scipy import spatial
from tqdm import tqdm


class GetExamples:
    def __init__(self, word, pickle, years):
        self.word = word
        self.pickle = pickle
        self.years = years

    def intersect_models(self, modeldict):
        _, _ = intersection_align_gensim(m1=modeldict[self.years[0]], m2=modeldict[self.years[1]])
        return modeldict

    def align_models(self, modeldict):
        _ = smart_procrustes_align_gensim(modeldict[self.years[0]], modeldict[self.years[1]])
        return modeldict

    @staticmethod
    def avg_feature_vector(sentence, model):
        num_features = model.vector_size
        words = [w for w in sentence if w in model]
        lw = len(words)
        if lw == 0:
            return None
        feature_matrix = np.zeros((lw, num_features), dtype='float32')
        for i in list(range(lw)):
            word = words[i]
            feature_matrix[i, :] = model[word]
        feature_vec = np.average(feature_matrix, axis=0)
        return feature_vec

    def create_examples(self, models, method):
        intersected_models = GetExamples.intersect_models(self, models)
        aligned_models = GetExamples.align_models(self, intersected_models)

        pickle = self.pickle

        old_contexts = list()
        new_contexts = list()

        base_years = list()
        new_years = list()

        word = self.word

        start = time.time()

        all_samples = {}

        log("Finding samples...")
        try:
            old_samples = pickle.get(self.years[0])
            new_samples = pickle.get(self.years[1])

        except KeyError:
            raise KeyError("Problem with", word, "because not enough samples found")

        model1 = aligned_models.get(self.years[0])
        model2 = aligned_models.get(self.years[1])

        # Keep matrices of sentence vectors for future usage:
        old_samples_vec = np.zeros((len(old_samples), model1.vector_size), dtype='float32')
        new_samples_vec = np.zeros((len(new_samples), model2.vector_size), dtype='float32')

        for nr, old_sample in enumerate(old_samples):
            old_sample_vec = GetExamples.avg_feature_vector(old_sample[0], model=model1)
            if old_sample_vec is not None:
                old_samples_vec[nr, :] = old_sample_vec

        for nr, new_sample in enumerate(new_samples):
            new_sample_vec = GetExamples.avg_feature_vector(new_sample[0], model=model2)
            if new_sample_vec is not None:
                new_samples_vec[nr, :] = new_sample_vec

        # Calculate all pairwise cosine distances at once:
        distances = spatial.distance.cdist(old_samples_vec, new_samples_vec, 'cosine')

        if method == 1:

            # Find the pair of most distant sentences:
            most_distant_ids = np.unravel_index(np.argmax(distances), distances.shape)

            # This is for debugging:

            # max_distance = np.max(distances)
            # most_distant_sentences = [old_samples[most_distant_ids[0]][1]]
            # new_samples[most_distant_ids[1]][1]]
            # print(most_distant_ids)
            # print(max_distance)
            # print(most_distant_sentences)

            # Reshaping most distant vectors a bit:
            vector0 = old_samples_vec[most_distant_ids[0]]
            vector0.shape = (1, model1.vector_size)
            vector1 = new_samples_vec[most_distant_ids[1]]
            vector1.shape = (1, model2.vector_size)

            # Now we calculate distances within time bins...
            old_distances = np.ravel(spatial.distance.cdist(vector0, old_samples_vec, 'cosine'))
            new_distances = np.ravel(spatial.distance.cdist(vector1, new_samples_vec, 'cosine'))

            # ...and five vectors nearest to the sentence vectors which was most distant
            # at the previous step. This vector itself is included in these 5, of course:
            old_nearest_ids = old_distances.argsort()[:6]
            new_nearest_ids = new_distances.argsort()[:6]

            # Extracting actual sentences corresponding to these vectors:
            five_old_samples = [old_samples[i][1] for i in old_nearest_ids]
            five_new_samples = [new_samples[i][1] for i in new_nearest_ids]

        elif method == 2:
            most_distant_ids = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
            old_samples_ids = set()
            new_samples_ids = set()
            for i in range(0, len(most_distant_ids)):
                old_samples_ids.add(most_distant_ids[0][i])
                new_samples_ids.add(most_distant_ids[1][i])
                if len(new_samples_ids) == 5:
                    break
            five_old_samples = [old_samples[i][1] for i in list(old_samples_ids)]
            five_new_samples = [new_samples[i][1] for i in list(new_samples_ids)]

        old_contexts.append(five_old_samples)
        new_contexts.append(five_new_samples)

        base_years.append(self.years[0])
        new_years.append(self.years[1])

        log("")
        log("This took ", format_time(time.time() - start))
        log("")
        output_df = pd.DataFrame({"WORD": word, "BASE_YEAR": base_years,
                                  "OLD_CONTEXTS": old_contexts, "NEW_YEAR": new_years,
                                  "NEW_CONTEXTS": new_contexts})
        output_df.index.names = ["ID"]
        output_df.to_csv('contexts_by_year.csv')
        log('Contexts saved to contexts_by_year.csv')

