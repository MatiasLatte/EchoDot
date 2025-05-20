from processing.hydro_capture import capture_and_log
import json
from pathlib import Path
import time


CONFIG_PATH = Path("../config.json")
TIMESTAMP_DIR = Path("timestamps")

with open(CONFIG_PATH) as f:
    config = json.load(f)

def passive_hearing() -> None:
    hydro = config["hydrophones"][0]
    hydro_id = hydro["id"]
    detection_status = capture_and_log(hydro_id)
    if detection_status:
        print(f"Detection from hydrophone {hydro_id}. Sending flag.")

def run():
    try:
        previous_time = 0
        while True:
            if time.time() - previous_time >= 2:
                passive_hearing()
                previous_time = time.time()
    except KeyboardInterrupt:
        print("Stopping passive hearing loop.")
