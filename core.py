from dataclasses import dataclass, field
from typing import List

import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


@dataclass
class FlightStatusLoadData:
    data_path: str
    preload_rows: int = 100_000
    features: List[str] = field(default_factory=lambda: [
        'FlightDate', 'Airline', 'Origin', 'Dest',
        'Cancelled', 'Diverted',
        'CRSDepTime', 'DepTime', 'DepDelayMinutes',
        'ArrDelayMinutes'
    ])


class FlightStatusPrediction:
    def __init__(self, data: FlightStatusLoadData):
        print("Load dataframe")
        self.load_data = data
        self.features = self.load_data.features
        self.dataframe = pd.read_csv(self.load_data.data_path, nrows=self.load_data.preload_rows)
        print("Dataframe loaded")

        self.dataframe_pca = None
        self.scaler = None
        self.pca_model = None
        self.feature_names = None

        self.X_lda = None
        self.y_lda = None
        self.lda_model = None
        self.lda_result = None

        self.df_efa = None
        self.fa_model = None
        self.loadings = None

    def preprocess_data(self):
        df = self.dataframe[self.features].copy()

        df['FlightDate'] = pd.to_datetime(df['FlightDate'])
        df['Month'] = df['FlightDate'].dt.month
        df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
        df = df.drop(columns=['FlightDate'])

        df = pd.get_dummies(df, columns=['Airline'], drop_first=True)

        df = df.drop(columns=['Origin', 'Dest'])

        df = df.dropna()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        self.dataframe_pca = df
        print(f"Data shape after preprocessing: {self.dataframe_pca.shape}")

    def build_pca(self):
        if self.dataframe_pca is None:
            self.preprocess_data()

        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.dataframe_pca)

        self.pca_model = PCA(n_components=2)
        principal_components = self.pca_model.fit_transform(scaled_data)

        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

        print("\n--- PCA Result ---")
        print(f"Explained Variance Ratio: {self.pca_model.explained_variance_ratio_}")

        components = pd.DataFrame(self.pca_model.components_, columns=self.dataframe_pca.columns, index=['PC1', 'PC2'])
        print("\nTop contributors to PC1:")
        print(components.loc['PC1'].abs().sort_values(ascending=False).head(3))

        self.dataframe_pca = pca_df
        return self

    def visualize_pca_results(self):
        if self.pca_model is None:
            print("Please run build_pca() first.")
            return None

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        sns.scatterplot(x='PC1', y='PC2', data=self.dataframe_pca, alpha=0.3, s=10)
        plt.title('PCA Scatter Plot (Flight Clusters)')
        plt.xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0] * 100:.1f}% Variance)')
        plt.ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1] * 100:.1f}% Variance)')

        plt.subplot(1, 3, 2)

        pc1_loadings = self.pca_model.components_[0]

        loadings_df = pd.DataFrame({'Feature': self.feature_names, 'Loading': pc1_loadings})
        loadings_df = loadings_df.sort_values(by='Loading', key=abs, ascending=False).head(10)

        sns.barplot(x='Loading', y='Feature', data=loadings_df, palette='coolwarm')
        plt.title('Top Features driving PC1 (Loadings)')
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)

        plt.subplot(1, 3, 3)
        y_pos = [0, 1]
        plt.bar(y_pos, self.pca_model.explained_variance_ratio_, alpha=0.7)
        plt.xticks(y_pos, ['PC1', 'PC2'])
        plt.title('Explained Variance Ratio')
        plt.ylabel('Percentage of Information')

        plt.tight_layout()
        plt.show()

        return self

    def preprocess_for_lda(self):
        print("\nPreprocessing for LDA...")

        df = self.dataframe[self.features].copy()

        def categorize_delay(minutes):
            if minutes < 15:
                return 'On Time'
            elif minutes < 60:
                return 'Small Delay'
            else:
                return 'Major Delay'

        df = df.dropna(subset=['ArrDelayMinutes'])

        y_labels = df['ArrDelayMinutes'].apply(categorize_delay)

        df = df.drop(columns=['ArrDelayMinutes', 'Cancelled', 'Diverted', 'Origin', 'Dest'])

        df['FlightDate'] = pd.to_datetime(df['FlightDate'])
        df['Month'] = df['FlightDate'].dt.month
        df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
        df = df.drop(columns=['FlightDate'])

        df = pd.get_dummies(df, columns=['Airline'], drop_first=True)

        df = df.dropna()

        y_labels = y_labels.loc[df.index]

        self.X_lda = df
        self.y_lda = y_labels

        print(f"LDA Features (X) shape: {self.X_lda.shape}")
        print(f"LDA Target (y) classes: {self.y_lda.unique()}")

    def build_lda(self):
        if self.X_lda is None:
            self.preprocess_for_lda()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_lda)

        self.lda_model = LinearDiscriminantAnalysis(n_components=2)
        X_lda_transformed = self.lda_model.fit_transform(X_scaled, self.y_lda)

        self.lda_result = pd.DataFrame(data=X_lda_transformed, columns=['LD1', 'LD2'])
        self.lda_result['Class'] = self.y_lda.values

        print("\n---  LDA Result ---")
        print(f"Explained Variance Ratio: {self.lda_model.explained_variance_ratio_}")
        return self

    def visualize_lda_results(self):
        if self.lda_result is None:
            print("Please run build_lda() first.")
            return

        plt.figure(figsize=(10, 8))

        custom_palette = {'On Time': 'green', 'Small Delay': 'orange', 'Major Delay': 'red'}

        sns.scatterplot(
            x='LD1', y='LD2',
            hue='Class',
            data=self.lda_result,
            palette=custom_palette,
            alpha=0.6,
            s=15
        )

        plt.title('LDA: Separation of Flight Delay Groups')
        plt.xlabel(f'LD1 ({self.lda_model.explained_variance_ratio_[0] * 100:.1f}%)')
        plt.ylabel(f'LD2 ({self.lda_model.explained_variance_ratio_[1] * 100:.1f}%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title='Delay Status')
        plt.show()
        return self

    def preprocess_for_efa(self):
        print("\n--- Preprocessing for EFA ---")
        df = self.dataframe[self.features].copy()

        cols_to_drop = ['FlightDate', 'Airline', 'Origin', 'Dest']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

        df = df.dropna()

        non_constant_cols = [col for col in df.columns if df[col].nunique() > 1]
        dropped_cols = set(df.columns) - set(non_constant_cols)
        if dropped_cols:
            print(f"Dropping constant columns (Zero Variance): {dropped_cols}")
            df = df[non_constant_cols]

        print(f"Columns used for EFA: {df.columns.tolist()}")

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        self.df_efa = df_scaled
        print(f"Data ready for Factor Analysis. Shape: {self.df_efa.shape}")

    def check_factorability(self):
        if self.df_efa is None:
            self.preprocess_for_efa()

        print("\n--- 1. Adequacy Tests ---")

        chi_square_value, p_value = calculate_bartlett_sphericity(self.df_efa)
        print(f"Bartlett’s Test p-value: {p_value} (Should be < 0.05)")

        kmo_all, kmo_model = calculate_kmo(self.df_efa)
        print(f"KMO Test Value: {kmo_model:.3f} (Should be > 0.6)")

        return self

    def build_efa(self, n_factors=3):
        if self.df_efa is None:
            self.preprocess_for_efa()

        print(f"\n--- 2. Building Factor Model ({n_factors} Factors) ---")

        self.fa_model = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
        self.fa_model.fit(self.df_efa)

        self.loadings = pd.DataFrame(
            self.fa_model.loadings_,
            index=self.df_efa.columns,
            columns=[f'Factor {i + 1}' for i in range(n_factors)]
        )

        variance = self.fa_model.get_factor_variance()
        var_df = pd.DataFrame(variance, index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],
                              columns=[f'Factor {i + 1}' for i in range(n_factors)])

        print("\nFactor Variance:")
        print(var_df)
        return self

    def visualize_efa(self):
        if self.loadings is None:
            print("Run build_efa() first.")
            return

        plt.figure(figsize=(8, 6))

        sns.heatmap(self.loadings, annot=True, cmap="coolwarm", center=0)

        plt.title('Factor Loadings (Correlations between Variables and Latent Factors)')
        plt.ylabel('Observed Variables')
        plt.xlabel('Latent Factors')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        ev, v = self.fa_model.get_eigenvalues()
        plt.scatter(range(1, self.df_efa.shape[1] + 1), ev)
        plt.plot(range(1, self.df_efa.shape[1] + 1), ev)
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue (Information)')
        plt.grid()
        plt.axhline(y=1, color='r', linestyle='--')
        plt.show()

    def evaluate_lda_performance(self, test_size=0.2):
        """
        Imparte datele in set de antrenament si testare, antreneaza un model temporar
        si afiseaza raportul de clasificare pentru a valida acuratetea.
        """
        if self.X_lda is None:
            self.preprocess_for_lda()

        print(f"\n--- EVALUARE MODEL (Split {100 - test_size * 100:.0f}/{test_size * 100:.0f}) ---")

        # 1. Split Date
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_lda, self.y_lda, test_size=test_size, random_state=42, stratify=self.y_lda
        )

        # 2. Scalare (Fit pe train, transform pe test pentru a evita data leakage)
        scaler_val = StandardScaler()
        X_train_scaled = scaler_val.fit_transform(X_train)
        X_test_scaled = scaler_val.transform(X_test)

        # 3. Antrenare Model de Validare
        lda_val = LinearDiscriminantAnalysis(n_components=2)
        lda_val.fit(X_train_scaled, y_train)

        # 4. Predictie
        y_pred = lda_val.predict(X_test_scaled)

        # 5. Metrici
        acc = accuracy_score(y_test, y_pred)
        print(f"Acuratete Globala pe setul de Test: {acc * 100:.2f}%")
        print("\nRaport Detaliat de Clasificare:")
        print(classification_report(y_test, y_pred))

        return acc
def do_pca():
    data_load_obj = FlightStatusLoadData(data_path="Zboruri_Sample_Proiect.csv")
    obj = FlightStatusPrediction(data_load_obj)
    obj.build_pca()\
        .visualize_pca_results()

def do_lda():
    data_load_obj = FlightStatusLoadData(data_path="Zboruri_Sample_Proiect.csv")
    obj = FlightStatusPrediction(data_load_obj)
    obj.build_lda()\
        .visualize_lda_results()


def do_efa():
    data_load_obj = FlightStatusLoadData(data_path="Zboruri_Sample_Proiect.csv")
    data_load_obj.features=[
        'FlightDate', 'Airline', 'Origin', 'Dest',
        'Cancelled', 'Diverted',
        'CRSDepTime', 'DepTime', 'DepDelayMinutes',
        'ArrDelayMinutes',
        'Distance', 'AirTime'
    ]
    obj = FlightStatusPrediction(data_load_obj)
    obj.build_efa()\
        .visualize_efa()

if __name__ == "__main__":
    # do_lda()
    do_pca()
    # do_efa()