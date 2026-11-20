# 1. Import Libraries: Bring in pandas to read and validate the dataset
import pandas as pd

# 2. Load the Raw Dataset: Read the CSV uploaded into tourism_project/data/
RAW_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(RAW_PATH)

# 3. Validate Columns: Ensure all expected schema columns exist before treating the dataset as registered
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# 4. Report a Summary: Print dimensions, column list, and target class counts & balance
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))

counts = df["ProdTaken"].value_counts()
pos_pct = (df["ProdTaken"].value_counts(normalize=True)[1] * 100).round(2)
print("ProdTaken counts:\n", counts)
print(f"ProdTaken positive represents {pos_pct}% of the data.")
