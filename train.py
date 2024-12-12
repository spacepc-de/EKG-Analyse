import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle  # Import für shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# 1. Daten laden

# Normale Herzschläge
normal_data = pd.read_csv("ptbdb_normal.csv", header=None)
normal_labels = np.zeros(len(normal_data))  # Label: 0 für normale Herzschläge

# Abnormale Herzschläge
abnormal_data = pd.read_csv("ptbdb_abnormal.csv", header=None)  # Ersetze mit deinem Datensatz
abnormal_labels = np.ones(len(abnormal_data))  # Label: 1 für abnormale Herzschläge

# Kombiniere die Daten
data = pd.concat([normal_data, abnormal_data])
labels = np.concatenate([normal_labels, abnormal_labels])

# Shuffle der Daten
data, labels = shuffle(data, labels)  # sklearn.utils.shuffle verwenden

# 2. Trainings-/Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape der Daten (falls notwendig)
X_train = np.expand_dims(X_train.values, axis=-1)
X_test = np.expand_dims(X_test.values, axis=-1)

# 3. Modell erstellen
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification: Normal (0) vs. Abnormal (1)
])

# 4. Modell kompilieren
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Modell trainieren
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 6. Modell evaluieren
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Optional: Modell speichern
model.save("ecg_abnormal_detection_model.h5")
