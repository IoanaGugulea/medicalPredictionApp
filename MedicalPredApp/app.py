import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Funcția pentru antrenarea modelelor
def train_and_select_best_model():
    data = pd.read_csv('medical_appointments.csv')
    data['Genul'] = data['Genul'].map({'Masculin': 0, 'Feminin': 1})
    data['PresiuneaSangelui'] = data['PresiuneaSangelui'].map({'scazut': 0, 'normal': 1, 'ridicat': 2})
    data['Diabet'] = data['Diabet'].map({'Nu': 0, 'Da': 1})
    
    X = data[['Varsta', 'IMC', 'Genul', 'PresiuneaSangelui', 'Diabet']]
    y = data['Rezultat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)

    if accuracy_knn > accuracy_nb:
        selected_model = model_knn
        best_accuracy = accuracy_knn
        model_type = 'kNN'
    else:
        selected_model = model_nb
        best_accuracy = accuracy_nb
        model_type = 'Naive Bayes'

    return selected_model, best_accuracy, model_type

# Funcția pentru antrenarea modelului
def train_model():
    selected_model, best_accuracy, model_type = train_and_select_best_model()

    data = pd.read_csv('medical_appointments.csv')
    data['Genul'] = data['Genul'].map({'Masculin': 0, 'Feminin': 1})
    data['PresiuneaSangelui'] = data['PresiuneaSangelui'].map({'scazut': 0, 'normal': 1, 'ridicat': 2})
    data['Diabet'] = data['Diabet'].map({'Nu': 0, 'Da': 1})

    X = data[['Varsta', 'IMC', 'Genul', 'PresiuneaSangelui', 'Diabet']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = selected_model
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Rutele Flask
@app.route('/')
def home():
    return render_template('index.html', model_type='kNN', accuracy=0.85)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        varsta = float(request.form['varsta'])
        imc = float(request.form['imc'])
        gen = 1 if request.form['gen'] == 'Feminin' else 0
        presiunea_sangelui = {'scazut': 0, 'normal': 1, 'ridicat': 2}[request.form['presiunea_sangelui']]
        diabet = 1 if request.form['diabet'] == 'Da' else 0

        input_data = np.array([[varsta, imc, gen, presiunea_sangelui, diabet]])

        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

        result = "Va veni la programare" if prediction == 1 else "Nu va veni la programare"
        return render_template('result.html', result=result)

if __name__ == '__main__':
    train_model()
    app.run(debug=True)
