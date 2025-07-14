import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
import pickle

# Load dataset
df = pd.read_csv("Air line passenger.csv")
df.drop(columns=["ID"], inplace=True)
df.dropna(inplace=True)

# Label encode target
target_col = 'Satisfaction'
df[target_col] = LabelEncoder().fit_transform(df[target_col])

# Balance the classes
df_majority = df[df[target_col] == 1]
df_minority = df[df[target_col] == 0]
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)
df = pd.concat([df_majority, df_minority_upsampled])

# Features
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
numerical_cols = [
    'Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
    'Departure and Arrival Time Convenience', 'On-board Service',
    'Seat Comfort', 'Leg Room Service', 'Cleanliness', 'Food and Drink',
    'In-flight Service', 'In-flight Wifi Service',
    'In-flight Entertainment', 'Baggage Handling',
    'Check-in Service', 'Ease of Online Booking', 'Gate Location', 'Online Boarding'
]

# Label encode categoricals
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Standardize numerical
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Train model
X = df[categorical_cols + numerical_cols]
y = df[target_col]
model = GradientBoostingClassifier()
model.fit(X, y)

# Save model and tools
with open("gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Model trained and saved.")
