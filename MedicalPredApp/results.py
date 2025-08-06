import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data = {
    'Varsta': [25, 45, 32, 50, 23, 60, 38, 41, 29, 55],
    'IMC': [22.5, 30.1, 28.4, 35.7, 18.6, 31.2, 26.5, 29.3, 24.1, 33.8],
    'Genul': ['Feminin', 'Masculin', 'Feminin', 'Masculin', 'Feminin', 'Masculin', 'Feminin', 'Masculin', 'Feminin', 'Masculin'],
    'PresiuneaSangelui': ['normal', 'ridicat', 'normal', 'ridicat', 'scazut', 'ridicat', 'normal', 'normal', 'scazut', 'ridicat'],
    'Diabet': ['Nu', 'Da', 'Nu', 'Da', 'Nu', 'Da', 'Nu', 'Da', 'Nu', 'Da'],
    'Rezultat': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]  # 1 - va veni la programare, 0 - nu va veni
}

# DataFrame creat dupa dictionarul de mai sus
df = pd.DataFrame(data)

# Preprocesare date
# encodare valorile categorice folosind LabelEncoder
label_encoder = LabelEncoder()

df['Genul'] = label_encoder.fit_transform(df['Genul'])
df['PresiuneaSangelui'] = label_encoder.fit_transform(df['PresiuneaSangelui'])
df['Diabet'] = label_encoder.fit_transform(df['Diabet'])

# Definire caracteristici și eticheta
X = df[['Varsta', 'IMC', 'Genul', 'PresiuneaSangelui', 'Diabet']]
y = df['Rezultat']

# impartirea datelor dupa un set de antrenament si test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# antrenarea unui model de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predictii
y_pred = model.predict(X_test)

# matricea de confuzie
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nu va veni', 'Va veni'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# indicatorii de performanta
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accurate: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
