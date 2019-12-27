#!/usr/bin/env python
# coding: utf-8

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plot
import numpy as np
from matplotlib import font_manager
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import configparser


config = configparser.RawConfigParser()
config.read('webvectors.cfg')

root = config.get('Files and directories', 'root')
path = config.get('Files and directories', 'font')
font = font_manager.FontProperties(fname=path)


def tsne_semantic_shifts(result, fname):
    """
    :result: a dictionary containing word_list and vector_list from aligned models
    """
    word_list = result["word_list"]
    vector_list = result["vector_list"]
    model_number = result["model_number"]

    perplexity = int(len(word_list) ** 0.5)

    embedding = TSNE(n_components=2, random_state=0, learning_rate=150, init="pca", perplexity=perplexity)
    np.set_printoptions(suppress=True)
    y = embedding.fit_transform(np.array(vector_list))

    word_coordinates = [y[i] for i in range(0, model_number)]
    x_coordinates, y_coordinates = y[:, 0], y[:, 1]

    plot.figure(figsize=(7, 7))
    plot.scatter(x_coordinates, y_coordinates)
    plot.axis("off")

    for label, x, y in list(zip(word_list, x_coordinates, y_coordinates))[
        :model_number
    ]:
        plot.annotate(
            label,
            xy=(x, y),
            weight="bold",
            xytext=(-len(label) * 4.5, 4),
            fontsize=12,
            textcoords="offset points",
        )

    for label, x, y in list(zip(word_list, x_coordinates, y_coordinates))[
        model_number:
    ]:
        plot.annotate(
            label, xy=(x, y), xytext=(-len(label) * 4.5, 4), textcoords="offset points"
        )

    plot.xlim(x_coordinates.min() - 10, x_coordinates.max() + 10)
    plot.ylim(y_coordinates.min() - 10, y_coordinates.max() + 10)

    for i in range(len(word_coordinates) - 1, 0, -1):
        plot.annotate(
            "",
            xy=(word_coordinates[i - 1][0], word_coordinates[i - 1][1]),
            weight="bold",
            xytext=(word_coordinates[i][0], word_coordinates[i][1]),
            arrowprops=dict(arrowstyle="-|>", color="indianred"),
        )
    plot.savefig(
        root + "data/images/tsne_shift/" + fname + ".png", dpi=150, bbox_inches="tight"
    )
    plot.close()
    plot.clf()


def pca_semantic_shifts(result, fname):
    """
    :result: a dictionary containing word_list and vector_list from aligned models
    """
    word_list = result["word_list"]
    vector_list = result["vector_list"]
    model_number = result["model_number"]

    embedding = PCA(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    y = embedding.fit_transform(np.array(vector_list))

    word_coordinates = [y[i] for i in range(0, model_number)]
    x_coordinates, y_coordinates = y[:, 0], y[:, 1]

    plot.figure(figsize=(10, 8))
    plot.scatter(x_coordinates, y_coordinates)
    plot.axis("off")

    for label, x, y in list(zip(word_list, x_coordinates, y_coordinates))[
        :model_number
    ]:
        plot.annotate(
            label,
            xy=(x, y),
            weight="bold",
            xytext=(-len(label) * 4, 4),
            fontsize=12,
            textcoords="offset points",
        )

    for label, x, y in list(zip(word_list, x_coordinates, y_coordinates))[
        model_number:
    ]:
        plot.annotate(
            label, xy=(x, y), xytext=(-len(label) * 4, 4), textcoords="offset points"
        )

    for i in range(len(word_coordinates) - 1, 0, -1):
        plot.annotate(
            "",
            xy=(word_coordinates[i - 1][0], word_coordinates[i - 1][1]),
            weight="bold",
            xytext=(word_coordinates[i][0], word_coordinates[i][1]),
            arrowprops=dict(arrowstyle="-|>", color="indianred"),
        )
    plot.savefig(
        root + "data/images/pca_shift/" + fname + ".png", dpi=150, bbox_inches="tight"
    )
    plot.close()
    plot.clf()


def singularplot(word, modelname, vector, fname):
    xlocations = np.array(list(range(len(vector))))
    plot.clf()
    plot.bar(xlocations, vector)
    plot_title = word.split('_')[0].replace('::', ' ') + '\n' + modelname + u' model'
    plot.title(plot_title, fontproperties=font)
    plot.xlabel('Vector components')
    plot.ylabel('Components values')
    plot.savefig(root + 'data/images/singleplots/' + modelname + '_' + fname + '.png', dpi=150,
                 bbox_inches='tight')
    plot.close()
    plot.clf()


def embed(words, matrix, classes, usermodel, fname, kind='TSNE'):
    perplexity = 6.0  # Should be smaller than the number of points!

    if kind.lower() == "tsne":
        embedding = TSNE(n_components=2, perplexity=perplexity, metric='cosine', n_iter=500,
                         init='pca')
    elif kind.lower() == "pca":
        embedding = PCA(n_components=2)
    else:
        raise ValueError("Kind is {}, must be TSNE or PCA".format(kind))

    y = embedding.fit_transform(matrix)

    print('2-d embedding finished', file=sys.stderr)

    class_set = [c for c in set(classes)]
    colors = plot.cm.rainbow(np.linspace(0, 1, len(class_set)))

    class2color = [colors[class_set.index(w)] for w in classes]

    xpositions = y[:, 0]
    ypositions = y[:, 1]
    seen = set()

    plot.clf()

    for color, word, class_label, x, y in zip(class2color, words, classes, xpositions, ypositions):
        plot.scatter(x, y, 20, marker='.', color=color,
                     label=class_label if class_label not in seen else "")
        seen.add(class_label)

        lemma = word.split('_')[0].replace('::', ' ')
        plot.annotate(lemma, xy=(x, y), xytext=(-len(lemma)*4.5, 0), textcoords="offset points",
                      size='x-large', weight='bold', fontproperties=font, color=color)

    plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plot.legend(loc='best')

    plot.savefig(root + 'data/images/' + kind.lower() + 'plots/' + usermodel + '_' + fname + '.png',
                 dpi=150, bbox_inches='tight')
    plot.close()
    plot.clf()
