import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def valori_proprii(valori, titlu='1) Varianța explicată'):
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


import matplotlib.pyplot as plt
import numpy as np


# --- 4. Cercul Corelațiilor Optimizat (C1 vs C2) ---
import matplotlib.pyplot as plt
import numpy as np


def desenare_cerc_pro(Rxc, varianta_explicata, variabile):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Fundal: Cercul unitate și axele
    cercuri = plt.Circle((0, 0), 1, color='#e0e0e0', fill=False, linewidth=2, linestyle='--')
    ax.add_artist(cercuri)
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)

    # 2. Definirea culorilor pe categorii (Punctualitate, Program, Logistică)
    # C1=Roșu, C2=Albastru, C3=Verde (conform importanței lor în heatmap)
    culori = ['#d63031', '#d63031', '#00b894', '#00b894', '#0984e3', '#0984e3']

    # 3. Trasarea vectorilor și a textului
    for i in range(len(variabile)):
        x, y = Rxc[i, 0], Rxc[i, 1]

        # Săgeata (lungimea indică calitatea reprezentării)
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.04,
                 length_includes_head=True, color=culori[i], alpha=0.9, linewidth=2)

        # Evitarea intercalării: plasăm textul mai departe de vârf (factor 1.15)
        # și adăugăm un fundal (bbox) pentru a fi vizibil peste alte linii
        ax.text(x * 1.15, y * 1.15, variabile[i],
                color='black', ha='center', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # 4. Configurări finale
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.grid(True, linestyle=':', alpha=0.4)

    # Etichete axe cu procentul de varianță explicată
    plt.xlabel(f'C1 - Punctualitate ({varianta_explicata[0] * 100:.2f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'C2 - Magnitudine Traseu ({varianta_explicata[1] * 100:.2f}%)', fontsize=12, fontweight='bold')
    plt.title('7) Cercul Corelațiilor (Proiecția Variabilelor pe C1 și C2)', fontsize=15, pad=20)

    # Notă explicativă jos
    plt.annotate('Săgețile lungi indică variabile bine reprezentate pe aceste axe',
                 xy=(0.5, -0.05), xycoords='axes fraction', ha='center',
                 fontsize=10, style='italic', color='#636e72')

    plt.tight_layout()
    plt.show()

def afisare():
    """Afișează toate ferestrele grafice active."""
    plt.show()