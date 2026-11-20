import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file = pd.read_csv("tirs.csv")

# Izvēlamies četrus kolonnas ar mērījumiem
columns = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

#X satur pazimes,y atbildi(prognozejam)
X = file[columns]
y = file["species"]

# 80% datu izmantojam apmācībai un 20% pārbaudei
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Izveidojam un apmācām modeli
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Prognozē testa datu pingvīnu sugas un aprēķinām precizitāti
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Aprēķinā bazes līnija
most_common_species = y_train.mode().iloc[0]
basic_score = accuracy_score(y_test,[most_common_species] * len(y_test))

print(f"Modeļa precizitāte: {accuracy*100}")
print(f"Bāzes līnija: {basic_score*100}")

# Salīdzinām modeļa un bāzes līnijas rezultātus
print(f"Starpība: {(accuracy - basic_score) * 100}")

if accuracy > basic_score:
    print("Secinājums: modelis pārspēj bāzes līniju")
else:
    print("Secinājums: modelis bāzes līniju nepārspēj")

# Nosak divas svarīgākās pazīmes
feature_importance = pd.Series(
    model.feature_importances_,
    index=columns
).sort_values(ascending=False)

print("Svarīgākās pazīmes:")
print(feature_importance.head(2))

# Atrodam testa rindas ar nepareizām prognozēm
error_mask = predictions != y_test
errors = X_test.loc[error_mask].copy()

errors["Patiesā atbilde"] = y_test.loc[errors.index]
errors["Prognoze"] = predictions[error_mask]

print("Kļūdaino prognožu skaits:", len(errors))

if len(errors) > 0:
    print(errors.head(5).to_string(index=False))
else:
    print("Testa datos kļūdainas prognozes netika atrastas.")