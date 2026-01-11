import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_flowchart_arrows():
    # Setăm dimensiunea figurii (mai înaltă pentru a avea loc de săgeți)
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')

    # Definim stilul cutiilor și al săgeților
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#003366', linewidth=2)

    # Coordonatele pe axa Y pentru fiecare pas (de sus în jos)
    # Lăsăm spațiu mai mare între ele pentru săgeți
    y_positions = [14, 11.5, 9, 6.5, 4, 1.5]

    steps = [
        "INPUT UTILIZATOR\n(Data, Companie, Ore, Întârziere)",
        "PREPROCESARE DATE\n(Extragere Lună/Zi & One-Hot Encoding)",
        "ALINIERE COLOANE (Reindex)\n(Adăugare coloane lipsă cu 0)",
        "SCALARE (StandardScaler)\n(Normalizarea valorilor)",
        "MODEL LDA (Predicție)\n(Calcul scor discriminant)",
        "OUTPUT FINAL\n(Clasa: On Time / Delay)"
    ]

    # Desenăm fiecare pas
    for i, (y, text) in enumerate(zip(y_positions, steps)):
        # 1. Desenăm cutia
        ax.text(5, y, text, ha='center', va='center', fontsize=11, fontweight='bold',
                color='#003366', bbox=box_props)

        # 2. Desenăm săgeata CĂTRE următorul pas (dacă nu e ultimul)
        if i < len(steps) - 1:
            # Coordonate start săgeată (baza cutiei curente)
            start_y = y - 0.6  # Ajustăm în funcție de înălțimea cutiei
            # Coordonate final săgeată (vârful cutiei următoare)
            end_y = y_positions[i + 1] + 0.6

            # Desenăm săgeata
            ax.annotate('', xy=(5, end_y), xytext=(5, start_y),
                        arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8))

    plt.title("Figura 4.2 - Fluxul Operațional al Simulării", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # Salvăm imaginea
    output_file = "diagrama_flux_sageti.png"
    plt.savefig(output_file, dpi=300)
    print(f"Diagrama a fost generată: {output_file}")


if __name__ == "__main__":
    draw_flowchart_arrows()