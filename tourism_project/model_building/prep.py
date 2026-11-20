import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")
df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True)
# Replacing mispelled  gender values as well as replacing MaritalStatus Unmarried Value with Single. During inference we will use dropdown box to ensure no misspelling
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken percentage distribution in train:")
print(ytrain.value_counts(normalize=True) * 100)
print("ProdTaken percentage distribution in test:")
print(ytest.value_counts(normalize=True) * 100)
