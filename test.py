import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

# 1. Datei laden
file_path = "mitbih_test.csv"  # Ersetze dies mit deinem Dateipfad
normal_data = pd.read_csv(file_path, header=None)

# 2. Daten vorbereiten
# Falls notwendig, erweitere die Dimensionen (Modell erwartet [samples, timesteps, features])
X_normal = np.expand_dims(normal_data.values, axis=-1)

y_normal = np.zeros(len(normal_data))  # Labels: 0 für normale Herzschläge

# 3. Modell laden
model = load_model("ecg_abnormal_detection_model.h5")

# 4. Vorhersagen erstellen
predictions = model.predict(X_normal)

# 5. Schwellenwert für binäre Klassifikation
threshold = 0.5
predicted_labels = (predictions > threshold).astype(int)

# 6. Ergebnisse auswerten
print("Classification Report:")
print(classification_report(y_normal, predicted_labels))

accuracy = accuracy_score(y_normal, predicted_labels)
print(f"Accuracy on PTBDB Normal Data: {accuracy:.4f}")
