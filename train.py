import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student_performance_prediction.csv")

# Remove Student ID
df = df.drop("Student ID", axis=1)

# Fill missing values
df["Attendance Rate"] = df["Attendance Rate"].fillna(
    df["Attendance Rate"].mean()
)

df["Previous Grades"] = df["Previous Grades"].fillna(
    df["Previous Grades"].mean()
)
df = df.dropna(subset=["Passed"])
# Convert text columns to numbers
le_extra = LabelEncoder()
le_parent = LabelEncoder()
le_passed = LabelEncoder()

df["Participation in Extracurricular Activities"] = le_extra.fit_transform(
    df["Participation in Extracurricular Activities"]
)

df["Parent Education Level"] = le_parent.fit_transform(
    df["Parent Education Level"]
)

df["Passed"] = le_passed.fit_transform(
    df["Passed"]
)
print("Parent Education Classes:")
print(le_parent.classes_)

print("Activities Classes:")
print(le_extra.classes_)
# Features and Target
X = df.drop("Passed", axis=1)
y = df["Passed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
print("\nTarget Distribution:")
print(df["Passed"].unique())
