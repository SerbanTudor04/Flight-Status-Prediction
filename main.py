from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FlightStatusLoadData:
    data_path: str
    preload_rows: int = 100_000
    features: List[str] = field(default_factory=lambda: [
        'FlightDate', 'Airline', 'Origin', 'Dest',
        'Cancelled', 'Diverted',
        'CRSDepTime', 'DepTime', 'DepDelayMinutes'
    ])


class FlightStatusPrediction:
    def __init__(self, data: FlightStatusLoadData):
        print("Load dataframe")
        self.load_data = data
        self.features = self.load_data.features
        self.dataframe = pd.read_csv(self.load_data.data_path, nrows=self.load_data.preload_rows)
        print("Dataframe loaded")


        # PCA Atributes
        self.dataframe_pca = None
        self.scaler = None
        self.pca_model = None
        self.feature_names = None

    def preprocess_data(self):
        """
        Converts non-numeric data to numeric so PCA can handle it.
        """
        # FIX: Changed self.raw_features to self.features
        df = self.dataframe[self.features].copy()

        # 2. Handle Dates
        df['FlightDate'] = pd.to_datetime(df['FlightDate'])
        df['Month'] = df['FlightDate'].dt.month
        df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
        df = df.drop(columns=['FlightDate'])

        # 3. Handle Categorical: Airline
        df = pd.get_dummies(df, columns=['Airline'], drop_first=True)

        # 4. Handle Categorical: Origin/Dest (Drop for now)
        df = df.drop(columns=['Origin', 'Dest'])

        # 5. Handle Missing Values
        df = df.dropna()

        # Ensure all data is numeric
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
        """
        Generates 3 key plots to understand the PCA analysis.
        """
        if self.pca_model is None:
            print("Please run build_pca() first.")
            return

        plt.figure(figsize=(18, 6))

        # --- PLOT 1: Scatter Plot of PC1 vs PC2 ---
        plt.subplot(1, 3, 1)
        sns.scatterplot(x='PC1', y='PC2', data=self.dataframe_pca, alpha=0.3, s=10)
        plt.title('PCA Scatter Plot (Flight Clusters)')
        plt.xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0] * 100:.1f}% Variance)')
        plt.ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1] * 100:.1f}% Variance)')

        # --- PLOT 2: Feature Loadings (What makes up PC1?) ---
        # This shows which original variables influence the First Component the most
        plt.subplot(1, 3, 2)

        # Get the contributions of each feature to PC1
        pc1_loadings = self.pca_model.components_[0]

        # Create a temporary DF to plot
        loadings_df = pd.DataFrame({'Feature': self.feature_names, 'Loading': pc1_loadings})
        loadings_df = loadings_df.sort_values(by='Loading', key=abs, ascending=False).head(10)

        sns.barplot(x='Loading', y='Feature', data=loadings_df, palette='coolwarm')
        plt.title('Top Features driving PC1 (Loadings)')
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)

        # --- PLOT 3: Explained Variance ---
        plt.subplot(1, 3, 3)
        y_pos = [0, 1]  # positions for PC1, PC2
        plt.bar(y_pos, self.pca_model.explained_variance_ratio_, alpha=0.7)
        plt.xticks(y_pos, ['PC1', 'PC2'])
        plt.title('Explained Variance Ratio')
        plt.ylabel('Percentage of Information')

        plt.tight_layout()
        plt.show()

        return self


if __name__ == "__main__":
    data_path = "data/Combined_Flights_2019.csv"
    data_load_obj = FlightStatusLoadData(data_path=data_path)
    obj = FlightStatusPrediction(data_load_obj)
    obj.build_pca()\
        .visualize_pca_results()