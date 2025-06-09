import joblib
import pandas as pd

def main():
    # Load the saved model
    model_file_path = 'Network_Threat_ML.joblib'
    
    try:
        model = joblib.load(model_file_path)
        print("Model loaded successfully.\n")
    except FileNotFoundError:
        print(f"Model file '{model_file_path}' not found. Please ensure the model is trained and saved.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # Define the feature names
    feature_names = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packets Length Total', 'Bwd Packets Length Total',
        'Fwd Packet Length Max', 'Fwd Packet Length Mean', 
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 
        'Bwd Packet Length Mean', 'Fwd Seg Size Min', 
        'Active Mean', 'Active Std', 'Active Max', 
        'Active Min', 'Idle Mean', 'Idle Std', 
        'Idle Max', 'Idle Min'
    ]

    # Collect input values from the user
    input_values = []
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter value for '{feature}': "))
                input_values.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    # Create a DataFrame for the input values
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Make a prediction using the model
    prediction = model.predict(input_df)

    # Print the predicted class label
    print(f"\nPredicted Class Label: {prediction[0]}")

if __name__ == '__main__':
    main()
