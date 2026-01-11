import numpy as np
import pandas as pd
import Function as f
import PCA as pca
import Graph as g
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 1. Citirea datelor
print("1) Citirea datelor de zbor pentru ACP")
tabel = pd.read_csv('dataIN/Zboruri_Sample_Proiect.csv', index_col=0)

# 2. Selecția variabilelor (cele 6 variabile originale)
vars_pca = [
    'DepDelayMinutes', 'DepDelay',
    'CRSDepTime', 'DepTime',
    'AirTime', 'Distance'
]

tabel_1=tabel.values
print(f"a) i) Numar obersavtii analizate initial: {tabel_1.shape[0]}")

# Curățăm datele (esențial pentru ACP)
tabel_clean = tabel[vars_pca].dropna()
X = tabel_clean.values
obs = tabel_clean.index.values

print(f"   ii) Număr observații analizate dupa eliminarea valorilor Nan: {X.shape[0]}")
print(f"b) Număr variabile analizate: {X.shape[1]}")
print()
# 3. Instanțiere model ACP (standardizare)
modelACP = pca.PCA(X)

# 4. Analiza Varianței si calcularea valorilor proprii
valori_proprii = modelACP.getAlpha()
varianta_explicata = valori_proprii / np.sum(valori_proprii)
varianta_cumulata = np.cumsum(varianta_explicata)

print()
print("Stabilirea pragului de retenție (80%)")
n_80 = np.where(varianta_cumulata >= 0.8)[0][0] + 1
print(f"Numar componente reținute (Criteriul Varianței): {n_80} componente")
print(f"Informație păstrată: {varianta_cumulata[n_80-1]*100:.2f}%\n")

nume_pc = [f'C{i+1}' for i in range(n_80)]


#A) Scoruri (Noile coordonate ale zborurilor în spațiul redus)
scoruri = modelACP.getScores()[:, :n_80]
pd.DataFrame(scoruri, index=obs, columns=nume_pc).to_csv("./dataOUT/Scores_ACP.csv")

#B) Factor Loadings (Corelațiile cu axele principale)
Rxc = modelACP.getRxc()[:, :n_80]
Rxc_df = pd.DataFrame(Rxc, index=vars_pca, columns=nume_pc)
Rxc_df.to_csv("./dataOUT/FactorLoadings_ACP.csv")

#C) Comunalități (Calitatea reprezentării variabilelor)
comun = modelACP.getComunalitati()[:, :n_80]
pd.DataFrame(comun, index=vars_pca, columns=nume_pc).to_csv("./dataOUT/Comunalitati_ACP.csv")


print()
print("3) Calculul Contributiilor variabilelor")
# --- 1. Calculul Contribuțiilor Variabilelor (%) ---
# Contribuția arată cât la sută din construcția unei axe aparține fiecărei variabile
contributii_var = (Rxc**2 / valori_proprii[:n_80]) * 100
contributii_df = pd.DataFrame(contributii_var, index=vars_pca, columns=nume_pc)
contributii_df.to_csv("./dataOUT/Contributii_Variabile_ACP.csv")
print(contributii_df)

# --- 2. Identificarea Outlierilor (Zborurile extreme pe C1 - Punctualitate) ---
# Extragem zborurile care au cele mai mici scoruri pe C1 (cele mai mari întârzieri)
scoruri_df = pd.DataFrame(scoruri, index=obs, columns=nume_pc)
top_outlieri_intarzieri = scoruri_df.sort_values(by='C1').head(10)
print("4) Top 10 zboruri cu cele mai mari abateri (Outlieri C1):")
print(top_outlieri_intarzieri)




#GRAFICE


#graficul valorilor proprii
g.valori_proprii(valori_proprii) #valori proprii

#corelograma gfactor loadings
g.corelograma(R2=Rxc_df, titlu='2) Corelograma Factor Loadings')


# graficul matricea scorurilor
plt.figure(figsize=(10, 7))
plt.scatter(scoruri[:, 0], scoruri[:, 1], alpha=0.5, c='royalblue', edgecolors='k')
plt.axhline(0, color='red', linestyle='--')
plt.axvline(0, color='red', linestyle='--')
plt.title("3) Matricea Scorurilor (Zborurile proiectate pe C1 și C2)")
plt.xlabel(f"C1 - Punctualitate/Întârzieri")
plt.ylabel(f"C2 - Magnitudine Zbor (Distanță)")
plt.grid(True)

# analiza t-SNE pentru clusterizarea observațiilor
if len(X) > 2000:
    indices = np.random.choice(len(X), 2000, replace=False)
    X_sample = X[indices]
    # Opțional: colorăm punctele în funcție de componenta C1 a acestor zboruri
    color_values = scoruri[indices, 0]
else:
    X_sample = X
    color_values = scoruri[:, 0]

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_values, cmap='RdYlGn', alpha=0.7, s=20)

plt.colorbar(scatter, label='Scor C1 (Punctualitate)')
plt.title('4) Vizualizare t-SNE: Clusterizarea naturală a zborurilor\n(Culorile indică gradul de punctualitate)', fontsize=14)
plt.xlabel('t-SNE Dimensiunea 1')
plt.ylabel('t-SNE Dimensiunea 2')
plt.grid(True, alpha=0.2)


#analiza t-SNE pentru clusterizarea observațiilor tu tot cu sageti
# t-SNE este foarte lent pe 48.000 de rânduri. Luăm un eșantion de 2000 pentru vizualizare clară.
if len(X) > 2000:
    indices = np.random.choice(len(X), 2000, replace=False)
    X_sample = X[indices]
    color_values = scoruri[indices, 0]  # Culorile vin din Scorurile C1 (Punctualitate)
    obs_sample = obs[indices]          # Datele zborurilor pentru eșantion
else:
    X_sample = X
    color_values = scoruri[:, 0]
    obs_sample = obs

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(12, 9))


scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_values,
                    cmap='RdYlGn', alpha=0.6, s=25, edgecolors='none')


df_tsne_tmp = pd.DataFrame({
    'TSNE1': X_tsne[:, 0],
    'TSNE2': X_tsne[:, 1],
    'C1': color_values
}, index=obs_sample)

top_outlieri_tsne = df_tsne_tmp.sort_values(by='C1').head(5)

for data_zbor, row in top_outlieri_tsne.iterrows():
    plt.annotate(
        f"OUTLIER C1\nData: {data_zbor}",
        xy=(row['TSNE1'], row['TSNE2']),
        xytext=(row['TSNE1'] + 10, row['TSNE2'] + 10), # Poziția textului
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle='arc3,rad=.1'),
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
    )

plt.colorbar(scatter, label='Performanță Punctualitate (Scor C1)')
plt.title('6) Vizualizare t-SNE: Clusterizarea naturală a zborurilor\n(Liniile indică zborurile cu cele mai mari abateri)')
plt.xlabel('t-SNE Dimensiunea 1')
plt.ylabel('t-SNE Dimensiunea 2')
plt.grid(True, alpha=0.1)
plt.show()




#calitatea reprezentarii variabilelor
comun_df = pd.DataFrame(data=comun, index=vars_pca, columns=nume_pc)

plt.figure(figsize=(10, 6))
ax = sns.heatmap(comun_df,
                 annot=True,
                 fmt=".3f",
                 cmap="YlGnBu",
                 linewidths=.5)

plt.title(f'5) Calitatea Reprezentării Variabilelor (Comunalități)\nRetenție: {n_80} Componente Principale',
          fontsize=14, pad=20)
plt.ylabel('Variabile Initiale', fontsize=12)
plt.xlabel('Componentele Principale', fontsize=12)

# Adăugăm o notă explicativă direct pe grafic
plt.annotate('Valori apropiate de 1.00 indică o reprezentare perfectă',
             xy=(0.5, -0.15), xycoords='axes fraction', ha='center',
             fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.show()




#contributia variabilelor in %
plt.figure(figsize=(10, 7))
sns.heatmap(contributii_df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5)
plt.title('5) Contribuția Variabilelor la Construcția Axelor (%)')


# cercul corelatiilor
g.desenare_cerc_pro(Rxc, varianta_explicata, vars_pca)

g.afisare()