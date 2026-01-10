import argparse
import pandas as pd
import sys
import os
import joblib  # <--- NEW IMPORT
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from core import FlightStatusPrediction, FlightStatusLoadData


class FlightDelayCLI:
    def __init__(self, data_path="Zboruri_Sample_Proiect.csv", model_file="flight_model.pkl"):
        self.data_path = data_path
        self.model_file = model_file

        self.lda_model = None
        self.scaler = None
        self.feature_columns = None

        self.model_wrapper = None

    def train(self):
        if os.path.exists(self.model_file):
            print(f"Found saved model: {self.model_file}")
            print("Loading model, scaler, and features")

            # Load the dictionary containing all artifacts
            artifacts = joblib.load(self.model_file)

            self.lda_model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.feature_columns = artifacts['features']

            print("Model loaded successfully.\n")
            return
        print(f"No saved model found. Loading data from {self.data_path}...")

        data_load = FlightStatusLoadData(data_path=self.data_path)
        print("Data loaded successfully.")
        print("Training model")

        self.model_wrapper = FlightStatusPrediction(data_load)
        self.model_wrapper.preprocess_for_lda()
        self.model_wrapper.build_lda()

        # Extract the artifacts we need to save
        self.lda_model = self.model_wrapper.lda_model
        self.feature_columns = self.model_wrapper.X_lda.columns

        # Fit the scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.model_wrapper.X_lda)

        # 3. SAVE THE MODEL
        print(f"Saving model to {self.model_file}...")
        artifacts = {
            'model': self.lda_model,
            'scaler': self.scaler,
            'features': self.feature_columns
        }
        joblib.dump(artifacts, self.model_file)

        print("Model trained and saved successfully.\n")

    def _prepare_input_vector(self, flight_date, airline, crs_dep, dep_time, delay_min):
        input_data = pd.DataFrame([{
            'FlightDate': flight_date,
            'Airline': airline,
            'CRSDepTime': crs_dep,
            'DepTime': dep_time,
            'DepDelayMinutes': delay_min
        }])

        input_data['FlightDate'] = pd.to_datetime(input_data['FlightDate'])
        input_data['Month'] = input_data['FlightDate'].dt.month
        input_data['DayOfWeek'] = input_data['FlightDate'].dt.dayofweek
        input_data = input_data.drop(columns=['FlightDate'])

        input_data = pd.get_dummies(input_data, columns=['Airline'])

        input_data = input_data.reindex(columns=self.feature_columns, fill_value=0)

        return self.scaler.transform(input_data)

    def predict_single(self, flight_date, airline, crs_dep, dep_time, delay_min):
        if self.lda_model is None:
            raise Exception("Model not trained or loaded. Run train() first.")

        input_scaled = self._prepare_input_vector(flight_date, airline, crs_dep, dep_time, delay_min)

        prediction_class = self.lda_model.predict(input_scaled)[0]
        probabilities = self.lda_model.predict_proba(input_scaled)[0]
        classes = self.lda_model.classes_

        return prediction_class, classes, probabilities

    def run_default_scenarios(self):
        scenarios = [
            {
                "Name": "Christmas Flight - Airline DL (Delta) - Storm Condition",
                "FlightDate": "2022-12-24",
                "Airline": "DL",
                "CRSDepTime": 1800,
                "DepTime": 1930,
                "DepDelayMinutes": 90
            },
            {
                "Name": "Christmas Flight - Airline WN (Southwest) - Normal Condition",
                "FlightDate": "2022-12-24",
                "Airline": "WN",
                "CRSDepTime": 1800,
                "DepTime": 1805,
                "DepDelayMinutes": 5
            }
        ]

        for sc in scenarios:
            print(f"--- Scenario: {sc['Name']} ---")
            pred_class, classes, probs = self.predict_single(
                sc["FlightDate"], sc["Airline"], sc["CRSDepTime"], sc["DepTime"], sc["DepDelayMinutes"]
            )

            print(f"Predicted Status: {pred_class}")
            print("Probabilities:")
            for cls, prob in zip(classes, probs):
                print(f"  - {cls}: {prob * 100:.2f}%")
            print("")

    def run_interactive_mode(self):
        print("\n--- Interactive Flight Status Prediction ---")
        print("Press Enter to accept the [default value].\n")

        default_date = datetime.today().strftime('%Y-%m-%d')
        date_input = input(f"Flight Date (YYYY-MM-DD) [{default_date}]: ").strip()
        flight_date = date_input if date_input else default_date

        default_airline = "TAROM"
        airline_input = input(f"Airline Code [{default_airline}]: ").strip()
        airline = airline_input if airline_input else default_airline

        default_crs = "1200"
        crs_input = input(f"Scheduled Departure (HHMM) [{default_crs}]: ").strip()
        crs_dep = int(crs_input) if crs_input else int(default_crs)

        default_dep = str(crs_dep)
        dep_input = input(f"Actual Departure (HHMM) [{default_dep}]: ").strip()
        dep_time = int(dep_input) if dep_input else int(default_dep)

        calc_delay = max(0, dep_time - crs_dep)
        if dep_time % 100 < crs_dep % 100:
            calc_delay = (dep_time - crs_dep) - 40

        delay_input = input(f"Departure Delay Minutes [{calc_delay}]: ").strip()
        delay_min = float(delay_input) if delay_input else float(calc_delay)

        print(f"\nAnalyzing flight: {airline} on {flight_date} (Delay: {delay_min} min)...")

        pred_class, classes, probs = self.predict_single(
            flight_date, airline, crs_dep, dep_time, delay_min
        )

        print(f"\nPREDICTED STATUS: {pred_class}")
        print("Confidence Levels:")
        for cls, prob in zip(classes, probs):
            marker = "<--" if cls == pred_class else ""
            print(f"    {cls:<12}: {prob * 100:05.2f}% {marker}")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Flight Delay Prediction CLI")

    parser.add_argument('--data', type=str, default="Zboruri_Sample_Proiect.csv", help="Path to the training CSV file")

    # NEW ARGUMENT: Allow user to specify a different model file name if they want
    parser.add_argument('--model', type=str, default="flight_model.pkl", help="Path to save/load the model file")

    subparsers = parser.add_subparsers(dest='command', required=True, help="Command to run")

    subparsers.add_parser('simulate', help="Run default comparison scenarios")

    subparsers.add_parser('interactive', help="Run in interactive Q&A mode")

    parser_pred = subparsers.add_parser('predict', help="Predict for a specific custom flight")
    parser_pred.add_argument('--date', type=str, required=True, help="Flight Date (YYYY-MM-DD)")
    parser_pred.add_argument('--airline', type=str, required=True, help="Airline Code")
    parser_pred.add_argument('--crs', type=int, required=True, help="Scheduled Time (HHMM)")
    parser_pred.add_argument('--actual', type=int, required=True, help="Actual Time (HHMM)")
    parser_pred.add_argument('--delay', type=float, required=True, help="Delay in Minutes")

    args = parser.parse_args()

    cli = FlightDelayCLI(data_path=args.data, model_file=args.model)
    try:
        cli.train()
    except FileNotFoundError:
        # Only errors out if training is required AND CSV is missing
        print(f"Error: Data file '{args.data}' not found and no saved model exists.")
        sys.exit(1)

    if args.command == 'simulate':
        cli.run_default_scenarios()

    elif args.command == 'interactive':
        cli.run_interactive_mode()

    elif args.command == 'predict':
        print(f"\n--- Custom Prediction for {args.airline} on {args.date} ---")
        pred_class, classes, probs = cli.predict_single(
            args.date, args.airline, args.crs, args.actual, args.delay
        )
        print(f"Predicted Status: {pred_class}")
        print("Probabilities:")
        for cls, prob in zip(classes, probs):
            print(f"  - {cls}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()