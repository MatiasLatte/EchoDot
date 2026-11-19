"""Local lake sound monitor. Run with --help for input options."""
import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter
from scipy.signal import resample_poly
from scipy.special import expit

HERE = Path(__file__).resolve().parent
RATE = 16000
WINDOW = 15600


class SoundFeatures:
    def __init__(self):
        self.model = Interpreter(model_path=str(HERE / "yamnet.tflite"), num_threads=2)
        self.model.allocate_tensors()
        self.input = self.model.get_input_details()[0]["index"]
        self.output = self.model.get_output_details()[0]["index"]

    def extract(self, audio, rate):
        divisor = math.gcd(rate, RATE)
        wave = resample_poly(audio, RATE // divisor, rate // divisor)
        if not len(wave):
            raise ValueError("Empty recording")
        # Remove recorder DC offset and reduce sensitivity to input gain.
        wave -= wave.mean()
        wave *= 0.2 / max(float(np.max(np.abs(wave))), 1e-4)
        wave = np.pad(wave, (0, max(0, WINDOW - len(wave))))
        starts = list(range(0, len(wave) - WINDOW + 1, WINDOW))
        if starts[-1] != len(wave) - WINDOW:
            starts.append(len(wave) - WINDOW)
        scores = []
        for start in starts:
            self.model.set_tensor(self.input, wave[start:start + WINDOW].astype(np.float32))
            self.model.invoke()
            scores.append(self.model.get_tensor(self.output)[0])
        return np.mean(scores, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wav", type=Path)
    mode.add_argument("--mic", action="store_true")
    mode.add_argument("--devices", action="store_true")
    parser.add_argument("--device", type=int, help="Hydrophone input number")
    args = parser.parse_args()
    if args.devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    extractor = SoundFeatures()
    with np.load(HERE / "classifier.npz") as weights:
        threshold = float(weights["threshold"])
        coef, bias = weights["coef"], weights["bias"]

    def report(audio, rate, seconds):
        score = expit(extractor.extract(audio, rate) @ coef + bias)
        print(f"{seconds:7.1f}s  {'possible boat' if score >= threshold else 'background'}  {score:.2f}", flush=True)

    if args.wav:
        with sf.SoundFile(args.wav) as source:
            for audio in source.blocks(blocksize=source.samplerate * 4, dtype="float32", always_2d=True):
                report(audio.mean(axis=1), source.samplerate, (source.tell() - len(audio)) / source.samplerate)
    else:
        import sounddevice as sd
        rate = int(sd.query_devices(args.device, "input")["default_samplerate"])
        seconds = 0
        with sd.InputStream(device=args.device, channels=1, samplerate=rate, dtype="float32") as stream:
            while True:
                audio, overflow = stream.read(rate * 4)
                if overflow:
                    print("Input overflow: audio skipped", flush=True)
                report(audio[:, 0], rate, seconds)
                seconds += 4


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
