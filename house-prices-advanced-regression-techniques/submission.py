import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

DATA_DIR = Path("./")  
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TotalSF"] = out.get("TotalBsmtSF", 0) + out.get("1stFlrSF", 0) + out.get("2ndFlrSF", 0)
    out["HouseAge"] = out.get("YrSold", 0) - out.get("YearBuilt", 0)
    out["SinceRemodel"] = out.get("YrSold", 0) - out.get("YearRemodAdd", 0)
    out["TotalBath"] = (
        out.get("FullBath", 0) +
        0.5 * out.get("HalfBath", 0) +
        out.get("BsmtFullBath", 0) +
        0.5 * out.get("BsmtHalfBath", 0)
    )
    out["Has2ndFlr"] = (out.get("2ndFlrSF", 0) > 0).astype(int)
    out["HasGarage"] = (out.get("GarageArea", 0) > 0).astype(int)
    out["HasBsmt"] = (out.get("TotalBsmtSF", 0) > 0).astype(int)
    out["HasFireplace"] = (out.get("Fireplaces", 0) > 0).astype(int)
    out["HasPool"] = (out.get("PoolArea", 0) > 0).astype(int)
    out["HasPorch"] = (
        out.get("OpenPorchSF", 0) + 
        out.get("EnclosedPorch", 0) + 
        out.get("3SsnPorch", 0) + 
        out.get("ScreenPorch", 0) > 0
    ).astype(int)
    return out

train_fe = engineer(train)
test_fe = engineer(test)

y = np.log1p(train_fe["SalePrice"].values)
X = train_fe.drop(columns=["SalePrice"])

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Ridge(alpha=1.0, random_state=42)

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

pipe.fit(X, y)
test_pred_log = pipe.predict(test_fe)
test_pred = np.expm1(test_pred_log)

test_pred = np.round(test_pred, 4)

submission = pd.DataFrame({
    "Id": test_fe["Id"],
    "SalePrice": test_pred
})
submission.to_csv(DATA_DIR / "submission.csv", index=False)

print("Submission file saved as submission.csv")
