import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict

import os
import imageio

from tqdm import tqdm


def create_gif(filenames, duration=1.0):
    gif_f = '../plots/KMeans/KMeans.gif'
    with imageio.get_writer(gif_f, mode='I', duration=duration) as writer:
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
        for f in filenames:
            os.remove(f)


class KMeans:
    def __init__(self, k):
        self.k = k

    def train(self, X, epochs, seed):
        N, M = X.shape
        np.random.seed(seed)
        # filenames = []

        self.X = X

        self.mi = X[np.random.choice(N, self.k, replace=False)]
        self.C = [np.ndarray((0, M)) for _ in range(self.k)]

        for e in range(epochs):
            d = np.linalg.norm(X[:, np.newaxis] - self.mi, axis=2)
            self.clusters = np.argmin(d, axis=1)
            for c in range(self.k):
                self.C[c] = np.array(np.where(self.clusters == c))
                self.mi[c] = np.mean(X[self.C[c]], axis=1)
            # if plot:
            #     filenames += [self.plot(e)]
        # if plot:
        #     create_gif(filenames)
        return self.clusters
        # print(C)

    def error(self):
        s = 0
        for x, c in zip(self.X, self.clusters):
            s += np.linalg.norm(x-self.mi[c])
        return s

    def plot(self, dataset):
        mi_df = pd.DataFrame({'x1': self.mi[:, 0], 'x2': self.mi[:, 1]})
        X_df = pd.DataFrame({'x1': self.X[:, 0], 'x2': self.X[:, 1], 'c': self.clusters})
        # palette = {0: 'red', 1: 'orange', 2: 'green', 3: 'blue', 4: 'violet', 5: 'brown', 6: 'black'}

        plt.figure()
        ax = sns.scatterplot(x=mi_df['x1'], y=mi_df['x2'], marker='X', s=500, hue=mi_df.index, palette='tab10')
        sns.scatterplot(x=X_df['x1'], y=X_df['x2'], ax=ax, hue=X_df['c'], palette='tab10')

        ax.set_xlabel('x1', fontsize=15)
        ax.set_ylabel('x2', fontsize=15)
        ax.set_title('Dane_2D_1', fontsize=20)
        ax.legend_.remove()
        plt.tight_layout()

        filename =  f'../plots/KMeans/{dataset}.png'
        plt.savefig(filename)
        plt.close()
        return filename


# class HGrouping:
#     def __init__(self, k):
#         self.X = None
#         self.c = None
#         self.k = k
#
#     def train(self, X, join_type='min'):
#         N, M = X.shape
#         self.X = X
#         self.c = np.arange(N)
#
#         # print(N-self.k-1)
#         # clusters = {x for x in range(N)}
#         for e in tqdm(range(N - self.k)):
#             c_dist = None
#             if join_type == 'min':
#                 c_dist = self.min_join()
#             elif join_type == 'max':
#                 c_dist = self.max_join()
#             elif join_type == 'mean':
#                 c_dist = self.mean_join()
#
#             # cd = np.where(self.c[:, None] != self.c, d, np.inf)
#             # c1, c2 = np.unravel_index(np.argmin(cd), d.shape)
#
#             min_c1, min_c2 = min(c_dist, key=c_dist.get)
#             self.c = np.where(self.c == self.c[min_c2], self.c[min_c1], self.c)
#
#         clusters = np.unique(np.sort(self.c))
#         value_to_rank = {value: rank for rank, value in enumerate(clusters)}
#         self.c = np.array([value_to_rank[value] for value in self.c])
#         return self.c
#         # print(self.c)
#
#     def min_join(self):
#         N, M = self.X.shape
#         d = np.linalg.norm(self.X[:, np.newaxis] - self.X, axis=2)
#
#         c_dist = defaultdict(lambda: np.inf)
#         for i in range(N):
#             for j in range(i):
#                 c1, c2 = sorted([self.c[i], self.c[j]])
#                 if c1 == c2:
#                     continue
#                 c_dist[(c1, c2)] = min(d[i, j], c_dist[(c1, c2)])
#         return c_dist
#
#     def max_join(self):
#         N, M = self.X.shape
#         d = np.linalg.norm(self.X[:, np.newaxis] - self.X, axis=2)
#
#         c_dist = defaultdict(lambda: 0)
#         for i in range(N):
#             for j in range(i):
#                 c1, c2 = sorted([self.c[i], self.c[j]])
#                 if c1 == c2:
#                     continue
#                 c_dist[(c1, c2)] = max(d[i, j], c_dist[(c1, c2)])
#         return c_dist
#
#     def mean_join(self):
#         N, M = self.X.shape
#         d = np.linalg.norm(self.X[:, np.newaxis] - self.X, axis=2)
#
#         c_dist = defaultdict(lambda: [0, 0])
#         for i in range(N):
#             for j in range(i):
#                 c1, c2 = sorted([self.c[i], self.c[j]])
#                 if c1 == c2:
#                     continue
#                 c_dist[(c1, c2)][0] += d[i, j]
#                 c_dist[(c1, c2)][1] += 1
#
#         return {k: v[0] / v[1] for k, v in c_dist.items()}


class HGrouping:
    def __init__(self, k):
        self.X = None
        self.c = None
        self.k = k

    def train(self, X, join_type='ward'):
        N, M = X.shape
        self.X = X
        self.c = np.arange(N)

        D = np.linalg.norm(self.X[:, np.newaxis] - self.X, axis=2) + np.diag(np.full(N, np.inf))
        clusters = [[i] for i in range(N)]

        for e in tqdm(range(N - self.k)):
            A_idx, B_idx = np.unravel_index(np.argmin(D), D.shape)
            K = D.shape[0]

            DC = np.full(K + 1, np.inf)
            for i in range(K):
                if i == A_idx or i == B_idx:
                    continue
                a_A, a_B, b, c = self.join(len(clusters[A_idx]), len(clusters[B_idx]), len(clusters[i]), join_type)
                DC[i] = (a_A * D[i, A_idx] + a_B * D[i, B_idx] +
                         b * D[A_idx, B_idx] + c * np.abs((D[i, A_idx] - D[i, B_idx])))
            DC[K] = np.inf
            DC = DC.reshape((K + 1, 1))

            D = np.append(D, DC[:K, :].T, axis=0)
            D = np.append(D, DC, axis=1)

            D = np.delete(D, [A_idx, B_idx], axis=0)
            D = np.delete(D, [A_idx, B_idx], axis=1)

            newC = clusters[A_idx] + clusters[B_idx]
            del clusters[max(A_idx, B_idx)]
            del clusters[min(A_idx, B_idx)]
            clusters += [newC]

        for i, c in enumerate(clusters):
            for p in c:
                self.c[p] = i

        return self.c

    def join(self, A, B, C, type):
        if type=='single':
            aA = 1/2
            aB = 1/2
            b = 0
            c = -1/2
            return aA, aB, b, c
        if type=='full':
            aA = 1/2
            aB = 1/2
            b = 0
            c = 1/2
            return aA, aB, b, c
        if type=='mean':
            aA = A / (A + B)
            aB = B / (A + B)
            b = 0
            c = 0
            return aA, aB, b, c
        if type=='centroid':
            aA = A / (A + B)
            aB = B / (A + B)
            b = -A * B / (A + B) ** 2
            c = 0
            return aA, aB, b, c
        if type == 'ward':
            aA = (A+C) / (A + B+C)
            aB = (B+C) / (A+B+C)
            b = -C / (A+B+C)
            c = 0
            return aA, aB, b, c
        raise Exception("type doesnt match")



class Spectral:
    def __init__(self, k):
        self.X = None
        self.k = k
        self.c = None

    def train(self, X, seed):
        N, M = X.shape
        self.X=X
        dist = np.linalg.norm(self.X[:, np.newaxis] - self.X, axis=2)
        self.c = np.zeros(N)

        A = numpy.zeros((N, N))
        for i in range(N):
            for j in range(i):
                A[i, j] = A[j, i] = np.exp(-dist[i, j])
        D = numpy.zeros((N, N))
        for i in range(N):
            D[i, i] = np.sum(A[:, i])

        L = D - A
        D_inv = numpy.zeros((N, N))
        for i in range(N):
            D_inv[i, i] = (1 / D[i, i]) ** 0.5
        L_norm = np.dot(np.dot(D_inv, L), D_inv)

        eigval, eigvec = numpy.linalg.eig(L_norm)
        eig_idx = numpy.argsort(eigval)[:self.k]
        X = eigvec[:, eig_idx]
        model = KMeans(self.k)
        self.c = model.train(X, 1000, seed)
        return self.c


def loss(y_true, y_pred):
    N = y_true.shape[0]
    y_true = y_true.reshape(N)

    correct = 0
    for i in range(N):
        for j in range(i):
            if (y_true[i] == y_true[j]) == (y_pred[i] == y_pred[j]):
                correct += 1
    return correct / (N * (N - 1) / 2)


def rp_loss(y_true, y_pred):
    N = y_true.shape[0]
    y_true = y_true.reshape(N)

    correct1, correct2 = 0, 0
    for i in range(N):
        if y_true[i] == y_pred[i]:
            correct1 += 1
        else:
            correct2 += 1
    return max(correct1, correct2) / N


def plot(X, c):
    X_df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'c': c})

    plt.figure()
    ax = sns.scatterplot(x=X_df['x1'], y=X_df['x2'], hue=X_df['c'], palette='tab10')

    ax.set_xlabel('x1', fontsize=15)
    ax.set_ylabel('x2', fontsize=15)
    ax.set_title('Dane_2D_1', fontsize=20)
    ax.legend_.remove()
    plt.tight_layout()
