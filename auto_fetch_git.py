import subprocess
import time
import threading
import os

def git_fetch():
    """Führt 'git fetch' aus und gibt das Ergebnis zurück."""
    try:
        result = subprocess.run(['git', 'fetch'], capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("Git fetch erfolgreich.")
            return True
        else:
            print(f"Git fetch Fehler: {result.stderr}")
            return False
    except Exception as e:
        print(f"Fehler beim Git fetch: {e}")
        return False

def auto_fetch_loop(interval_minutes=1):
    """Schleife für automatisches Fetching alle X Minuten."""
    while True:
        git_fetch()
        time.sleep(interval_minutes * 60)  # Warte X Minuten

if __name__ == "__main__":
    print("Starte Auto-Fetch für Git (alle 5 Minuten). Drücke Ctrl+C zum Stoppen.")
    # Starte in einem separaten Thread, damit das Skript nicht blockiert
    fetch_thread = threading.Thread(target=auto_fetch_loop, args=(5,))
    fetch_thread.daemon = True
    fetch_thread.start()

    # Halte das Hauptprogramm am Laufen
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Auto-Fetch gestoppt.")