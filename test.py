import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Loading data for baseline ML model...")
df = pd.read_csv('MMORPG_Trades_Cleaned.csv')

# 1. Prepare Tabular Features
# Standard ML models cannot read Player IDs or network structures easily.
# We will only give it the transaction amount and the trade type to prove that 
# "Amount alone is not enough" to detect complex fraud.
X = pd.get_dummies(df[['In_Game_Currency_Value', 'Trade_Type']]) 
y = df['Is_Fraudulent_Trade']

# 2. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train a standard Random Forest Classifier
print("Training standard Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    min_samples_leaf=1, 
    max_features='sqrt', 
    random_state=42
)
rf_model.fit(X_train, y_train)

# 4. Make predictions and evaluate
y_pred = rf_model.predict(X_test)

print("\n--- Baseline Model Performance (Standard ML) ---")
# We use classification_report to get Precision, Recall, and F1-Score as required by your rubric
print(classification_report(y_test, y_pred, zero_division=0))