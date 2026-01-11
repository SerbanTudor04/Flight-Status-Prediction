import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("Zboruri_Sample_Proiect.csv")

cols_to_keep = ['DepDelay', 'TaxiOut', 'Distance', 'CRSDepTime', 'Month', 'DayOfWeek', 'ArrDelay']
df_lda = df[cols_to_keep].copy()
df_lda = df_lda.dropna()

def clasifica_intarziere(minute):
    if minute < 15:
        return 0
    elif 15 <= minute < 60:
        return 1
    else:
        return 2

# 0 = OnTime
# 1 = Small Delay
# 2 = Major Delay

df_lda['DelayGroup'] = df_lda['ArrDelay'].apply(clasifica_intarziere)

print(df_lda)

sns.set_theme(style="whitegrid")


plt.figure(figsize=(8, 8))
counts = df_lda['DelayGroup'].value_counts()
plt.pie(counts, labels=['On Time (0)', 'Small Delay (1)', 'Major Delay (2)'],
        autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ffcc99','#ff9999'])
plt.title('Distribuția Claselor de Întârziere în Setul de Date')
plt.show()


plt.figure(figsize=(10, 8))
correlation_matrix = df_lda.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matricea de Corelație a Variabilelor')
plt.show()

X = df_lda.drop('DelayGroup', axis=1)
y = df_lda['DelayGroup']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Set Antrenare: {X_train.shape}")
print(f"Set Testare: {X_test.shape}")

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

y_pred = lda.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acuratețea modelului: {acc:.2%}")

print("\nRaport de Clasificare:")
print(classification_report(y_test, y_pred, target_names=['OnTime', 'Small Delay', 'Major Delay']))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: OnTime', 'Pred: Small', 'Pred: Major'],
            yticklabels=['Real: OnTime', 'Real: Small', 'Real: Major'])
plt.title('Matricea de Confuzie - Performanța LDA')
plt.ylabel('Clasa Reală')
plt.xlabel('Clasa Prezumată de Model')
plt.show()