import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

train_df = pd.read_csv("train.csv")  
test_df = pd.read_csv("test.csv")

combined_df = pd.concat([train_df.drop(columns='y'), test_df], axis=0)

categorical_cols = combined_df.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    label_encoders[col] = le

train_encoded = combined_df.iloc[:len(train_df)]
test_encoded = combined_df.iloc[len(train_df):]

# Separate features and labels
X_train = train_encoded.drop(columns='id')
y_train = train_df['y']
X_test = test_encoded.drop(columns='id')
test_ids = test_df['id']

# LightGBM model
model = LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred_probs = model.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({
    'id': test_ids,
    'y': y_pred_probs.round(1)
})

submission_df.to_csv("submission.csv", index=False)

print("Submission file saved as 'submission.csv'")
