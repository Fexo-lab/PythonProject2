import webview
import subprocess
import time
import os

from config import STREAMLIT_PORT, STREAMLIT_HOST, UI_WIDTH, UI_HEIGHT, UI_TITLE, STREAMLIT_STARTUP_DELAY

# Pfad zur app_ui.py
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_file = os.path.join(current_dir, "app_ui.py")

# Streamlit im Hintergrund starten (Headless Mode)
proc = subprocess.Popen([
    "streamlit", "run", ui_file,
    "--server.headless", "true",
    "--server.port", str(STREAMLIT_PORT)
])

# Dem Server Zeit geben, um hochzufahren
time.sleep(STREAMLIT_STARTUP_DELAY)

try:
    # Das native Fenster öffnen
    window = webview.create_window(
        UI_TITLE,
        f'http://{STREAMLIT_HOST}:{STREAMLIT_PORT}',
        width=UI_WIDTH,
        height=UI_HEIGHT,
        resizable=True,
        confirm_close=True
    )

    webview.start()

finally:
    # Sauberes Beenden
    proc.terminate()
    print("Trading Bot wurde sicher beendet.")