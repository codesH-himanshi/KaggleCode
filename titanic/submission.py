import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

train_df = pd.read_csv("train.csv")    
test_df = pd.read_csv("test.csv")

drop_cols = ['Name', 'Cabin', 'Ticket']
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# missing values
imputer_age = SimpleImputer(strategy='median')
imputer_fare = SimpleImputer(strategy='median')
imputer_embarked = SimpleImputer(strategy='most_frequent')

train_df['Age'] = imputer_age.fit_transform(train_df[['Age']])
test_df['Age'] = imputer_age.transform(test_df[['Age']])

train_df['Fare'] = imputer_fare.fit_transform(train_df[['Fare']])
test_df['Fare'] = imputer_fare.transform(test_df[['Fare']])

train_df['Embarked'] = imputer_embarked.fit_transform(train_df[['Embarked']]).ravel()
test_df['Embarked'] = imputer_embarked.transform(test_df[['Embarked']]).ravel()

# Encode categorical variables
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# data for model
X_train = train_df.drop(columns=['Survived', 'PassengerId'])
y_train = train_df['Survived']
X_test = test_df.drop(columns=['PassengerId'])
test_ids = test_df['PassengerId']

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission_df = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})

submission_df.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' generated.")
