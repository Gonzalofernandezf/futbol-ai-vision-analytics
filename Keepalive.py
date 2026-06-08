import pyautogui
import time
from datetime import datetime

pyautogui.FAILSAFE = False

INTERVAL_SEC = 600          # cada 45s (más conservador para sesiones largas)
JITTER_PX = 5

print(f"🟢 Keepalive activo. Movimiento cada {INTERVAL_SEC}s.")
print(f"   Inicio: {datetime.now().strftime('%H:%M:%S')}")
print("   Dejá el ratón sobre área vacía de la pestaña Colab.")
print("   Ctrl+C aquí cuando vuelvas.\n")

count = 0
try:
    while True:
        x, y = pyautogui.position()
        pyautogui.moveTo(x + JITTER_PX, y, duration=0.1)
        pyautogui.moveTo(x, y, duration=0.1)
        count += 1
        if count % 20 == 0:  # log cada ~15 min
            print(f"   [{datetime.now().strftime('%H:%M:%S')}] {count} pings enviados")
        time.sleep(INTERVAL_SEC)
except KeyboardInterrupt:
    print(f"\n🛑 Detenido tras {count} pings.")