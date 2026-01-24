# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ------------------------------
# Step 1: Load data
# ------------------------------
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Original shape:", df.shape)

# ------------------------------
# Step 2: Drop unnecessary columns
# ------------------------------
columns_to_remove = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
df.drop(columns=columns_to_remove, inplace=True)
print("Columns removed. Remaining shape:", df.shape)

# ------------------------------
# Step 3: Map target column
# ------------------------------
df['Attrition'] = df['Attrition'].map({"Yes": 1, "No": 0})
print("Attrition value counts:\n", df['Attrition'].value_counts())

# ------------------------------
# Step 4: Feature engineering
# ------------------------------
df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['LongCommute'] = (df['DistanceFromHome'] > 10).astype(int)

# ------------------------------
# Step 5: Split features & target
# ------------------------------
X = df.drop("Attrition", axis=1)
y = df['Attrition']

# ------------------------------
# Step 6: Identify categorical & numerical columns
# ------------------------------
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# ------------------------------
# Step 7: Preprocessing pipeline
# ------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ------------------------------
# Step 8: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Step 9: Fit-transform training data & transform test data
# ------------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Train shape:", X_train_processed.shape)
print("Test shape:", X_test_processed.shape)
print("Preprocessing completed!")

