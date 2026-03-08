import webview
import subprocess
import time
import sys
import os
import threading

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

def auto_fetch_loop(interval_minutes=5):
    """Schleife für automatisches Fetching alle X Minuten."""
    while True:
        git_fetch()
        time.sleep(interval_minutes * 60)  # Warte X Minuten

# 1. Den Pfad zur app_ui.py finden
# Das stellt sicher, dass das Skript die Datei auch findet, wenn du es aus verschiedenen Ordnern startest
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_file = os.path.join(current_dir, "app_ui.py")

# 2. Streamlit im Hintergrund starten (Headless Mode)
# Wir fügen '--server.port 8501' hinzu, um sicherzugehen, dass es immer auf demselben Port läuft
proc = subprocess.Popen([
    "streamlit", "run", ui_file,
    "--server.headless", "true",
    "--server.port", "8501"
])

# Dem Server 3-5 Sekunden Zeit geben, um hochzufahren
time.sleep(4)

# Starte Auto-Fetch Thread
fetch_thread = threading.Thread(target=auto_fetch_loop, args=(5,))
fetch_thread.daemon = True
fetch_thread.start()

try:
    # 3. Das native Fenster öffnen
    # Wir zeigen auf localhost:8501, wo unser Streamlit-Server jetzt läuft
    window = webview.create_window(
        'AI Quantum Trader Pro v1.0',
        'http://localhost:8501',
        width=1280,
        height=900,
        resizable=True,
        confirm_close=True # Fragt nach, ob man wirklich beenden will
    )

    webview.start()

finally:
    # 4. Sauberes Beenden: Wenn das Fenster geschlossen wird, killen wir den Hintergrund-Server
    proc.terminate()
    print("Trading Bot wurde sicher beendet.")