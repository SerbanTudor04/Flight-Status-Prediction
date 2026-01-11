import numpy as np
import pandas as pd
import PCA as pca
import Graph as g
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 1. Citirea datelor
tabel = pd.read_csv('dataIN/Zboruri_Sample_Proiect.csv', index_col=0)

# 2. Curățare și selecție automată a variabilelor numerice
tabel_numeric = tabel.select_dtypes(include=[np.number])
print(f"a) i) Numar observații analizate inițial: {tabel_numeric.shape[0]}")

# Excludem coloanele care nu sunt variabile de performanță (ID-uri, ani, coduri geografice)
cols_to_exclude = [col for col in tabel_numeric.columns if any(x in col for x in ['ID', 'SeqID', 'Year', 'Fips', 'Wac'])]
tabel_filtrat = tabel_numeric.drop(columns=cols_to_exclude)

# Păstrăm doar variabilele care au variație și eliminăm rândurile cu Nan
tabel_clean = tabel_filtrat.loc[:, tabel_filtrat.std() > 0.01].dropna()

X = tabel_clean.values
obs = tabel_clean.index.values
nume_variabile = tabel_clean.columns.tolist()

print(f"   ii) Număr observații după eliminarea valorilor Nan: {X.shape[0]}")
print(f"b) Număr variabile analizate: {X.shape[1]}")
print()

# 3. Execuție model ACP
modelACP = pca.PCA(X)
valori_proprii = modelACP.getAlpha()

# Corecție pentru stabilitate matematică (protecție împotriva valorilor negative infime)
valori_proprii[valori_proprii < 1e-7] = 1e-7

# 4. Analiza Varianței și Stabilirea Pragului de Retenție (80%)
varianta_explicata = valori_proprii / np.sum(valori_proprii)
varianta_cumulata = np.cumsum(varianta_explicata)

print("Stabilirea pragului de retenție (80%)")
# Căutăm indexul unde varianța cumulată atinge pragul de 0.8
n_80 = np.where(varianta_cumulata >= 0.8)[0][0] + 1

print(f"Numar componente reținute (Criteriul Varianței): {n_80} componente")
print(f"Informație păstrată: {varianta_cumulata[n_80-1]*100:.2f}%\n")

nume_pc_retinute = [f'C{i+1}' for i in range(n_80)]

# 5. Extragere și Salvare Rezultate (Asigură-te că folderul dataOUT există deja!)
print(f"Salvare rezultate ({n_80} componente) în folderul dataOUT...")

# a) Factor Loadings
Rxc = modelACP.getRxc()[:, :n_80]
Rxc_df = pd.DataFrame(data=Rxc, index=nume_variabile, columns=nume_pc_retinute)
Rxc_df.to_csv("./dataOUT/FactorLoadings_ACP.csv")

# b) Scoruri
scoruri = modelACP.getScores()[:, :n_80]
scoruri_df = pd.DataFrame(data=scoruri, index=obs, columns=nume_pc_retinute)
scoruri_df.to_csv("./dataOUT/Scoruri_ACP.csv")

# c) Comunalități
comun = modelACP.getComunalitati()[:, :n_80]
comun_df = pd.DataFrame(data=comun, index=nume_variabile, columns=nume_pc_retinute)
comun_df.to_csv("./dataOUT/Comunalitati_ACP.csv")

# 6. Generare Grafice
print("Generare vizualizări grafice...")










# Scree Plot
g.valori_proprii(valori_proprii)

# Corelograma Loadings
g.corelograma(R2=Rxc_df, titlu=f'Corelograma ACP (Retenție {n_80} PC)')

# Matricea Comunalităților
g.intensitate_legaturi(x=comun_df, titlu=f'Matricea Comunalităților ({n_80} PC)')

# Matricea Scorurilor (Scatter Plot)
plt.figure(figsize=(10, 7))
plt.scatter(scoruri[:, 0], scoruri[:, 1], alpha=0.5, edgecolors='k', c='royalblue')
plt.axhline(0, color='red', linestyle='--')
plt.axvline(0, color='red', linestyle='--')
plt.title("Matricea Scorurilor (Proiecția zborurilor pe C1 și C2)")
plt.xlabel("C1 - Componenta Principală 1")
plt.ylabel("C2 - Componenta Principală 2")
plt.grid(True)











# --- 5. d) Calculul Contribuțiilor Variabilelor (%) ---
# Contribuția arată ponderea fiecărei variabile în formarea unei componente
contributii_var = (Rxc**2 / valori_proprii[:n_80]) * 100
contributii_df = pd.DataFrame(contributii_var, index=nume_variabile, columns=nume_pc_retinute)
contributii_df.to_csv("./dataOUT/Contributii_Variabile_ACP.csv")

# --- 5. e) Identificarea Outlierilor (Zborurile extreme pe C1) ---
# Extragem primele 10 zboruri cu cele mai mici scoruri pe C1 (cele mai problematice)
top_outlieri = scoruri_df.sort_values(by='C1').head(10)
top_outlieri.to_csv("./dataOUT/Top_10_Outlieri_C1.csv")


# Heatmap pentru Comunalități (Calitatea reprezentării)
plt.figure(figsize=(12, 8))
sns.heatmap(comun_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
plt.title(f'Calitatea Reprezentării Variabilelor (Comunalități)\nRetenție: {n_80} PC', fontsize=14)

# Heatmap pentru Contribuții (%)
plt.figure(figsize=(12, 8))
sns.heatmap(contributii_df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5)
plt.title('Contribuția Variabilelor la Construcția Axelor (%)', fontsize=14)

# --- 7. Cercul Corelațiilor (Correlation Circle) ---
# Reparăm eroarea: adăugăm .subplots()
fig, ax = plt.subplots(figsize=(10, 10))

# Trasăm cercul unitate
cercuri = plt.Circle((0,0), 1, color='#e0e0e0', fill=False, linewidth=2, linestyle='--')
ax.add_artist(cercuri)
plt.axhline(0, color='black', linewidth=1, alpha=0.5)
plt.axvline(0, color='black', linewidth=1, alpha=0.5)

# Paletă de culori pentru cele 28 de variabile
colors = plt.cm.get_cmap('tab20', len(nume_variabile))

for i, var in enumerate(nume_variabile):
    x, y = Rxc[i, 0], Rxc[i, 1]
    # Desenăm doar variabilele care au o reprezentare minimă pentru a nu aglomera graficul
    if np.sqrt(x**2 + y**2) > 0.3:
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.04, length_includes_head=True,
                 color=colors(i), alpha=0.8, linewidth=1.5)
        ax.text(x * 1.1, y * 1.1, var, color='black', ha='center', va='center',
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.grid(True, linestyle=':', alpha=0.4)
plt.xlabel(f'C1 ({varianta_explicata[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'C2 ({varianta_explicata[1]*100:.2f}%)', fontsize=12)
plt.title('Cercul Corelațiilor (Primele 2 Componente)', fontsize=14, pad=20)

# --- 8. Analiza Finală a Outlierilor ---
# Extragem zborurile cele mai atipice pe prima componentă
print("\nIdentificare Outlieri (Top 10 zboruri problematice pe C1):")
outlieri_C1 = scoruri_df.sort_values(by='C1').head(10)
print(outlieri_C1)
outlieri_C1.to_csv("./dataOUT/Outlieri_Critici_C1.csv")

plt.show()





g.afisare()