from dataclasses import dataclass, field
from typing import List

import StandardScaler
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@dataclass
class FlightStatusLoadData:
    data_path: str
    preload_rows: int = 100_000
    # Use field for mutable default arguments in dataclasses to avoid issues
    features: List[str] = field(default_factory=lambda: [
        'FlightDate', 'Airline', 'Origin', 'Dest',
        'Cancelled', 'Diverted',
        'CRSDepTime', 'DepTime', 'DepDelayMinutes'
    ])


class FlightStatusPrediction:
    def __init__(self,data:FlightStatusLoadData):
        print("Load dataframe")
        self.load_data =data
        self.features = self.load_data.features
        self.dataframe = pd.read_csv(self.load_data.data_path,nrows=self.load_data.preload_rows)
        print("Dataframe loaded")

        self.dataframe_pca = None
        self.scaler = None

    def build_pca(self):
        # Fac o copie a DF ului numai pentru coloanele de care am nevoie
        self.dataframe_pca=self.dataframe[self.features].copy()
        self.dataframe_pca.dropna(inplace=True)
        print(f"Data shape after cleaning: {self.dataframe_pca.shape}")

        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.dataframe_pca)





if __name__ == "__main__":
    # Dataset from https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022/data
    data_path = "data/Combined_Flights_2019.csv"

    data_load_obj = FlightStatusLoadData(data_path=data_path)

    obj = FlightStatusPrediction(data_load_obj)
    obj.build_pca()

