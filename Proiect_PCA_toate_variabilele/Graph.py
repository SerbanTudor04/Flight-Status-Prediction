import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def valori_proprii(valori, titlu='Scree Plot - Varianța explicată'):
    """
    Reprezintă grafic valorile proprii pentru a identifica 'cotul' și pragul de variație.
    """
    plt.figure(num=titlu, figsize=(10, 7))
    plt.title(titlu, fontsize=14)
    plt.xlabel("Componente principale", fontsize=12)
    plt.ylabel("Valori proprii (Alpha)", fontsize=12)

    componente = ['C' + str(i + 1) for i in range(valori.shape[0])]

    # Plotarea valorilor proprii
    plt.plot(componente, valori, marker='o', color='b', linestyle='-')

    # Adăugarea unei linii pentru Criteriul Kaiser (Valoarea 1)
    plt.axhline(y=1, color='r', linestyle='--', label='Prag Kaiser (1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)


def corelograma(R2, titlu='Corelograma Factor Loadings', color='RdBu'):
    """
    Afișează corelațiile dintre variabilele originale și componentele principale.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(R2, annot=True, cmap=color, vmin=-1, vmax=1, center=0)
    plt.title(titlu)


def intensitate_legaturi(x, titlu='Harta Termică', color='Reds'):
    """
    Reprezintă grafic intensitatea legăturilor (pentru scoruri sau comunalități).
    """
    plt.figure(figsize=(12, 9))
    sns.heatmap(x, cmap=color, annot=False)
    plt.title(titlu)


def afisare():
    """Afișează toate ferestrele grafice active."""
    plt.show()