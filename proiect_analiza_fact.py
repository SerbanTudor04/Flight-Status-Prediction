import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import factor_analyzer as fa
import seaborn as sb
import matplotlib.pyplot as plt

# --- PAS 1 - Incarcarea, selectia si curatarea datelor ---

# Citim un esantion de 50.000 de randuri
#cale_fisier = 'Combined_Flights_2022.csv'
#df = pd.read_csv(cale_fisier).sample(n=50000, random_state=42) #random-state: la fiecare rurale raman aceleasi date
#zb_total = pd.read_csv(cale_fisier)
#print(len(zb_total))  #4078318 de inregistrari

#Lista coloane
# header_df = pd.read_csv(cale_fisier, nrows=0)
# lista_coloane = header_df.columns.values.tolist()
# print(lista_coloane)

#1. Incarcam datele
cale_fisier = 'dataIN_AF/Zboruri_Sample_Proiect.csv'

zboruri_df = pd.read_csv(cale_fisier)
zboruri_df_copy = zboruri_df.copy()

# zboruri_df.to_csv('Zboruri_Sample_Proiect.csv', index=False)
# print("Fisierul 'Zboruri_Sample_Proiect.csv' a fost creat pentru echipa")

# Am ales variabilele numerice care descriu diferite etape ale zborului
col_selectate = [
    'DepDelayMinutes',
    'ArrDelayMinutes',
    'AirTime',
    'ActualElapsedTime',
    'Distance'
]

# Curatam df
zboruri_df = zboruri_df[col_selectate].dropna()

# Matrice X si etichete
X = zboruri_df.values
obs = zboruri_df.index.values
cols_nume = zboruri_df.columns.values

# print(f"Date incarcate si curatate. Numar observatii: {len(obs)}, Numar variabile: {len(cols_nume)}")
# print("Primele 5 randuri din matricea de lucru:")
# print(zboruri_df.head())


# --- PAS 2: Standardizarea si Testele de Validare ---

# 1. Standardizez matricea X
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Salvare X_std in DataFrame pentru a pastra etichetele (util pentru pasii viitori)
X_std_df = pd.DataFrame(data=X_std, index=obs, columns=cols_nume)
X_std_df.to_csv('dataOUT_AF/X_std.csv')

# 2. Testul de sfericitate Bartlett
# Verifica daca matricea de corelatie este diferita de o matrice identitate
sfericitate_Bartlett = fa.calculate_bartlett_sphericity(x=X_std)
print('\n',sfericitate_Bartlett, type(sfericitate_Bartlett))
if sfericitate_Bartlett[0] > sfericitate_Bartlett[1]:
    print('\nExista cel putin un factor comun!')
else:
    print('\nNu exista factori comuni!')
    exit(-1)

# 3. Indicele Kaiser-Meyer-Olkin (KMO)
# Masoara adecvarea datelor pentru analiza factoriala
KMO = fa.calculate_kmo(x=X_std)
print(KMO, type(KMO))
if KMO[1] > 0.5:
    print('\nVariabilele observate pot fi exprimate prin factori!')
else:
    print('\nNu exista factori care sa exprime variabilele initiale!')
    exit(-2)

# Reprezantare grafica a indicilor KMO
vector_kmo = KMO[0]
print('\n',vector_kmo, type(vector_kmo), vector_kmo.shape)
matrice_kmo = vector_kmo[:, np.newaxis]
print('\n',matrice_kmo, type(matrice_kmo), matrice_kmo.shape)

matrice_kmo_df = pd.DataFrame(data=matrice_kmo, index=col_selectate, columns=['Indici KMO'])


def afisare_intensitate_kmo(matrice, titlu='1. Indici Kaiser-Meyer-Olkin'):
    plt.figure(num=titlu, figsize=(10, 6))
    plt.title(label=titlu, fontsize=14, color='Blue', pad=20)
    sb.heatmap(data=np.round(matrice,decimals=2), cmap='inferno', annot=True,
               vmin=0, vmax=1)
    plt.show()

afisare_intensitate_kmo(matrice_kmo_df)


# --- PAS 3: Identificarea numarului de factori ---
#1. Initializez modelul fara rotatie pentru a calcula valorile proprii
# fa_init = fa.FactorAnalyzer(rotation=None)
# fa_init.fit(X_std)

#1. Calculez valorile proprii
matrice_cor = np.corrcoef(X_std, rowvar=False)

ev,v = np.linalg.eig(matrice_cor)

ev = np.sqrt(ev)

#2. Extrag valorile proprii (eigenvalues)
# ev,v = fa_init.get_eigenvalues()
print("\nValorile proprii (Eigenvalues):")
for i, val in enumerate(ev):
    print(f"Factor {i+1}: {val:.4f}")
factori = [f"Factor {i+1}" for i in range(len(ev))]

#3. Scree plot
def afisare_scree_plot(ev, titlu="2. Scree Plot"):
    plt.figure(figsize=(10, 6))
    plt.title(label=titlu, fontsize=14, color='Blue', pad=20)
    plt.scatter(range(1,len(ev)+1), ev, color='DeepPink', s=100, zorder=5)
    plt.plot(range(1,len(ev)+1), ev, color='Magenta', linewidth=2, zorder=4)

    plt.xticks(range(1,len(ev)+1), factori, fontsize=12)
    for i, val in enumerate(ev):
        plt.text(i+1, val + 0.08, f"({val:.2f})",
                 ha='center', va='bottom',
                 fontsize=14, color='DarkSlateGrey')

    #plot Kaiser
    plt.axhline(y=1, color='Black', linestyle='--', label='Prag Kaiser(val-1)')

    #etichete axe
    plt.xlabel('Numar factor', fontsize=12)
    plt.ylabel('Valoare proprie (Eigenvalue)', fontsize=12)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

afisare_scree_plot(ev)


# --- PAS 4: Model - 2 factori cu rotatie varimax ---
fa_final = fa.FactorAnalyzer(n_factors=2, rotation='varimax')
fa_final.fit(X_std)

#factory loadings
loadings = fa_final.loadings_
loadings_df=pd.DataFrame(data=loadings, index=col_selectate, columns=['Factor 1', 'Factor 2'])
print("\nMatrice loadings dupa rotatie:")
print(loadings_df)


def afisare_loadings(df, titlu="3. Matrice Factory Loadings"):
    plt.figure(num=titlu, figsize=(8, 6))
    plt.title(titlu, fontsize=14, color='Blue', pad=20)
    sb.heatmap(df, cmap='RdPu', annot=True, vmin=-1, vmax=1)
    plt.show()

afisare_loadings(loadings_df)

# --- PAS 5: Model - Scoruri factoriale ---
scoruri = fa_final.transform(X_std)

scoruri_df = pd.DataFrame(data=scoruri, index=obs, columns=["Scor_Dimensiune_Zbor", 'Scor_Intarzieri'])

df_final_scoruri = zboruri_df.join(scoruri_df)

df_final_scoruri.to_csv("dataOUT_AF/Rezultate_Analiza_Factoriala.csv")
print(df_final_scoruri.head(), type(df_final_scoruri))

def afisare_scoruri(df_scoruri, titlu="4. Harta Zboruri: Dimensiune vs Intarzieri"):
    plt.figure(num=titlu, figsize=(12,8))
    plt.title(titlu, fontsize=15, color='Blue', pad=20)
    sb.scatterplot(data=df_scoruri, x='Scor_Dimensiune_Zbor', y='Scor_Intarzieri', alpha=0.4, color="Magenta")

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.xlabel('F1: Dimensiune Zbor (Scurt -> Lung)', fontsize=12)
    plt.ylabel('F2: Performanta Zbor (La timp -> Intarziat)', fontsize=12)

    plt.grid(alpha=0.2)
    plt.show()

afisare_scoruri(scoruri_df)

#Care sunt primele 10 cele mai intarziate zboruri?
zboruri_df_copy = zboruri_df_copy.join(scoruri_df)
top_10_zboruri_intarziate = zboruri_df_copy.sort_values(by="Scor_Intarzieri", ascending=False)[:10]

cols_vizualizare = [
    'FlightDate',
    'Marketing_Airline_Network',
    'OriginCityName',
    'DestCityName',
    'DepDelayMinutes',
    'ArrDelayMinutes',
    'AirTime',
    'ActualElapsedTime',
    'Scor_Intarzieri',
]

print(top_10_zboruri_intarziate[cols_vizualizare])
top_10_zboruri_intarziate[cols_vizualizare].to_csv('dataOUT_AF/Top_10_zboruri_intarziate.csv', index=False)
