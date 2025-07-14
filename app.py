from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model + tools
with open("gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

categorical_cols = list(label_encoders.keys())
numerical_cols = [
    'Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
    'Departure and Arrival Time Convenience', 'On-board Service',
    'Seat Comfort', 'Leg Room Service', 'Cleanliness', 'Food and Drink',
    'In-flight Service', 'In-flight Wifi Service', 'In-flight Entertainment',
    'Baggage Handling', 'Check-in Service', 'Ease of Online Booking',
    'Gate Location', 'Online Boarding'
]
all_features = categorical_cols + numerical_cols

sample_data = passenger_data =  {
    'Gender': 'Female',
    'Age': 35,
    'Customer Type': 'Returning',
    'Type of Travel': 'Business',
    'Class': 'Business',
    'Flight Distance': 1200,
    'Departure Delay': 0,
    'Arrival Delay': 0,
    'Departure and Arrival Time Convenience': 5,
    'On-board Service': 5,
    'Seat Comfort': 5,
    'Leg Room Service': 5,
    'Cleanliness': 5,
    'Food and Drink': 5,
    'In-flight Service': 5,
    'In-flight Wifi Service': 5,
    'In-flight Entertainment': 5,
    'Baggage Handling': 5,
    'Check-in Service': 5,
    'Ease of Online Booking': 5,
    'Gate Location': 5,
    'Online Boarding': 5
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probabilities = None

    if request.method == "POST":
        form_data = request.form

        if "suggest" in form_data:
            return render_template("index.html", suggest=sample_data)

        input_data = []
        for col in all_features:
            val = form_data.get(col)
            if val is None or val == '':
                val = 0
            elif col in categorical_cols:
                le = label_encoders[col]
                if val in le.classes_:
                    val = le.transform([val])[0]
                else:
                    print(f"⚠️ Unknown label '{val}' for column '{col}'. Defaulting to 0.")
                    val = 0
            else:
                val = float(val)
            input_data.append(val)

        input_df = pd.DataFrame([input_data], columns=all_features)
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        result = "✅ Satisfied" if prediction == 1 else "❌ Not Satisfied"
        probabilities = f"Satisfied: {round(prob[1]*100, 2)}% | Not Satisfied: {round(prob[0]*100, 2)}%"

        print("\n🎯 User Input:")
        print(input_df)

    return render_template("index.html", result=result, suggest=None, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True)
