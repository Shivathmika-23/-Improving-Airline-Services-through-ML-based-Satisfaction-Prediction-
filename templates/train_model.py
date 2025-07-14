import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load your dataset
df = pd.read_csv("Air line passenger.csv")

# Drop unnecessary columns
df.drop(['ID'], axis=1, inplace=True)

# Drop missing values (if any)
df.dropna(inplace=True)

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
target_col = 'Satisfaction'
numerical_cols = [col for col in df.columns if col not in categorical_cols + [target_col]]

# Label encode categorical columns
label_encoders = {}
for col in categorical_cols + [target_col]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    if col != target_col:
        label_encoders[col] = le

# Scale numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split features and target
X = df[categorical_cols + numerical_cols]
y = df[target_col]

# Train model
model = GradientBoostingClassifier()
model.fit(X, y)

# Save model, scaler, and encoders
with open("gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model, scaler, and encoders saved.")
