import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load your dataset
df = pd.read_csv("Air line passenger.csv")

# Columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
numerical_cols = [
    'Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
    'Departure and Arrival Time Convenience', 'On-board Service',
    'Seat Comfort', 'Leg Room Service', 'Cleanliness', 'Food and Drink',
    'In-flight Service', 'In-flight Wifi Service',
    'In-flight Entertainment', 'Baggage Handling'
]

# Drop ID and target
df = df.drop(['ID', 'Satisfaction'], axis=1)

# Label encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Fit scaler
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save scaler and encoders
with open("scaler_encoder.pkl", "wb") as f:
    pickle.dump((scaler, label_encoders), f)

print("Scaler and encoders saved.")
