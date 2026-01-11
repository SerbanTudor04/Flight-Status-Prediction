import numpy as np


def standardizare(X):
    medii = np.mean(a=X, axis=0)
    abateri_standard = np.std(a=X, axis=0)
    abateri_standard[abateri_standard == 0] = 1
    return (X - medii) / abateri_standard