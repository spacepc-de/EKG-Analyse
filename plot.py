import matplotlib.pyplot as plt
import csv

# Dateiname der CSV-Datei
csv_filename = "ptbdb_abnormal.csv"
# Nummer des anzuzeigenden Herzschlags (beginnend bei 0)
heartbeat_number = 5

def plot_csv_data(filename, heartbeat_number):
    try:
        # Daten aus der CSV-Datei lesen
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = []
            for row in reader:
                # Konvertiere Werte von wissenschaftlicher Notation zu Float und speichere jede Zeile separat
                heartbeat = [float(value) for value in row]
                data.append(heartbeat)

        # Prüfen, ob die gewünschte Herzschlag-Nummer gültig ist
        if heartbeat_number < 0 or heartbeat_number >= len(data):
            print(f"Ungültige Herzschlag-Nummer: {heartbeat_number}. Verfügbar sind 0 bis {len(data)-1}.")
            return

        # Erstellen des Plots für den angegebenen Herzschlag
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(data[heartbeat_number])), data[heartbeat_number], label=f"Herzschlag {heartbeat_number+1}", color="blue")

        plt.title(f"EKG-Graph - Herzschlag {heartbeat_number+1}")
        plt.xlabel("Zeit (ms)")
        plt.ylabel("Amplitude (mV)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Plot anzeigen
        plt.show()

    except FileNotFoundError:
        print(f"Datei {filename} wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

# Aufruf der Funktion
plot_csv_data(csv_filename, heartbeat_number)