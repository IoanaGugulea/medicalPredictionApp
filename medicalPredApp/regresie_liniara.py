import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Citește fișierul CSV cu datele PM2.5 pentru zona Ilfov
csv_filename = "pm25_data.csv"
dates = []
pm25_values = []

# Citim datele din fișier
with open(csv_filename, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            # Convertim datele pentru PM2.5
            pm25 = float(row["PM2.5 (µg/m³)"])
            
            # Adăugăm valorile PM2.5 în lista pm25_values
            pm25_values.append(pm25)
        except ValueError:
            continue  # Ignoră rândurile cu date invalide

# Creăm un array pentru X (timpul sau indecșii zilelor)
X = np.arange(len(pm25_values)).reshape(-1, 1)  # Folosim indexul zilelor ca variabilă independentă
y = np.array(pm25_values)  # Valorile PM2.5 ca variabilă dependentă

# Împărțim datele în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creăm și antrenăm modelul de regresie liniară
model = LinearRegression()
model.fit(X_train, y_train)

# Facem predicții pe setul de testare
y_pred = model.predict(X_test)

# Evaluăm modelul folosind eroarea pătratică medie (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Eroarea Pătratică Medie (MSE): {mse}')

# Plotează datele reale și valorile prezise
plt.figure(figsize=(12, 6))

# Plotează valorile reale PM2.5
plt.plot(X, y, label='Date reale PM2.5', color='blue')

# Plotează predicțiile (valori prezise de regresia liniară)
plt.scatter(X_test, y_pred, label='Predicții PM2.5', color='red')

# Setăm etichetele pentru axele X și Y
plt.xlabel('Zile', fontsize=14)
plt.ylabel('PM2.5 (µg/m³)', fontsize=14)
plt.title('Predictia valorilor PM2.5 pentru zona Ilfov prin Regresie Liniară', fontsize=18)
plt.legend()

# Afișăm graficul
plt.tight_layout()
plt.show()

# Salvăm coeficientul și interceptul regresiei într-un fișier
with open("regression_results.txt", "w") as f:
    f.write(f"Coeficientul regresiei (panta): {model.coef_[0]}\n")
    f.write(f"Interceptul regresiei: {model.intercept_}\n")
# Predicții pentru zile viitoare
future_days = np.arange(len(pm25_values), len(pm25_values) + 7).reshape(-1, 1)  # 7 zile viitoare
future_predictions = model.predict(future_days)

# Afișează predicțiile pentru zilele viitoare
print("Predicții pentru următoarele 7 zile:", future_predictions)
