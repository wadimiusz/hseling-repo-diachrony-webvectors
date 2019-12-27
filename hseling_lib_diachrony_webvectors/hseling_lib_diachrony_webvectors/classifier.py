from algos import GlobalAnchors, Jaccard, KendallTau
from gensim.models import KeyedVectors

from sklearn.linear_model import LogisticRegression
from typing import Iterable, Tuple
from sklearn.exceptions import NotFittedError
import functools
from sklearn.preprocessing import StandardScaler
import numpy as np


@functools.lru_cache(maxsize=-1)
def get_algo_by_kind_and_two_models(kind: str, model1: KeyedVectors,
                                    model2: KeyedVectors):
    if kind.lower() == "global_anchors":
        return GlobalAnchors(model1, model2)
    elif kind.lower() == "jaccard":
        return Jaccard(model1, model2, top_n_neighbors=50)
    elif kind.lower() == "kendall_tau":
        return KendallTau(model1, model2, top_n_neighbors=50)



class ShiftClassifier:
    def __init__(self):
        self.clf = LogisticRegression(class_weight='balanced')
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: Iterable[Tuple[str, KeyedVectors, KeyedVectors]],
            y: Iterable[float]):
        X_processed, y_processed = list(), list()
        for (word, model1, model2), label in zip(X, y):
            try:
                features = self.feature_extract(word, model1, model2)
                X_processed.append(features)
                y_processed.append(label)
            except KeyError:
                pass

        X_processed = self.scaler.fit_transform(X_processed)
        self.clf.fit(X=X_processed, y=y_processed)
        self.fitted = True
        return self

    def predict(self, X: Iterable[Tuple[str, KeyedVectors, KeyedVectors]]):
        if not self.fitted:
            raise NotFittedError

        X_processed = list()
        for word, model1, model2 in X:
            features = self.feature_extract(word, model1, model2)
            X_processed.append(features)
        X_processed = self.scaler.transform(X_processed)
        return self.clf.predict(X_processed)

    def predict_proba(self, X: Iterable[Tuple[str, KeyedVectors, KeyedVectors]]):
        if not self.fitted:
            raise NotFittedError

        X_processed = list()
        for word, model1, model2 in X:
            features = self.feature_extract(word, model1, model2)
            X_processed.append(features)
        X_processed = self.scaler.transform(X_processed)
        return self.clf.predict_proba(X_processed)[:, 1]

    def feature_extract(self, word, model1, model2):
        if word not in model1.vocab:
            raise KeyError("Word {} is not in the "
                           "vocab of model1".format(word))

        if word not in model2.vocab:
            raise KeyError("Word {} is not in the "
                           "vocab of model1".format(word))

        procrustes_score = model1[word] @ model2[word]  # models have previously been aligned with Procrustes analysis

        global_anchors_score = \
            get_algo_by_kind_and_two_models(
                "global_anchors", model1, model2).get_score(word)

        jaccard_score = get_algo_by_kind_and_two_models("jaccard", model1, model2).get_score(word)

        kendall_tau_score = get_algo_by_kind_and_two_models("kendall_tau", model1, model2).get_score(word)
        features = [procrustes_score, global_anchors_score, jaccard_score,
                    kendall_tau_score]
        return features