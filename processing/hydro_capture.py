import shutil
from datetime import datetime, UTC
from pathlib import Path
from processing.classifier import classify_and_handle
import random

USE_TEST_AUDIO = True  # TODO: Flip to False when using real hydrophones

def capture_and_log(hydro_id: str) -> bool:
    timestamp = datetime.now(UTC).timestamp()
    filename = f"buffer_{timestamp}.wav"
    filepath = f"buffer/{filename}"
    if USE_TEST_AUDIO:
        record_audio_test(filename)
    else:
        record_audio(filename)
    return classify_and_handle(filepath=filepath, hydro_id=hydro_id, timestamp=timestamp)


def record_audio(filename: str) -> None:
    Path("buffer").mkdir(exist_ok=True)
    open(f"buffer/{filename}", "wb").close()  # Mock recording


def record_audio_test(filename: str) -> None:
    Path("buffer").mkdir(exist_ok=True)
    sample_files = list(Path("test_audio").glob("*.wav"))
    if not sample_files:
        raise FileNotFoundError("No .wav files found in 'test_audio/'. Add test samples.")
    selected = random.choice(sample_files)
    shutil.copy(selected, f"buffer/{filename}")
    print(f"[Mock Record] Using test audio: {selected.name}")
