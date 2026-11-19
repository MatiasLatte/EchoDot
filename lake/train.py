"""Fit a boat/background classifier on a reproducible sample of local WAVs."""
import argparse
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from listen import HERE, SoundFeatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="AWhispernetData directory")
    args = parser.parse_args()
    rng = random.Random(42)
    extractor = SoundFeatures()
    train, validation, test = [], [], []
    for label, folder in enumerate(("ambient_ocean", "prop/oceanship_complete")):
        groups = {}
        for path in sorted((args.root / folder).glob("*.wav")):
            date = re.search(r'(\d{8})T', path.name)
            if date:
                groups.setdefault(date[1], []).append(path)
        if len(groups) < 4:
            raise ValueError(f"Need multiple recording days in {folder}")
        dates = sorted(groups)
        rng.shuffle(dates)
        split = max(1, len(dates) // 4)
        for days, limit, rows in [(dates[:split], 200, test), (dates[split:2 * split], 200, validation),
                                  (dates[2 * split:], 800, train)]:
            files = [path for day in days for path in groups[day]]
            for path in rng.sample(files, min(limit, len(files))):
                audio, rate = sf.read(path, dtype="float32", always_2d=True)
                rows.append((extractor.extract(audio.mean(axis=1), rate), label))
    features, labels = zip(*train)
    model = LogisticRegression(C=10, max_iter=1000).fit(features, labels)
    features, labels = zip(*validation)
    probabilities = model.predict_proba(features)[:, 1]
    # Bound background alerts on separate calibration days.
    threshold = max(0.5, np.quantile(probabilities[np.asarray(labels) == 0], 0.95))
    features, labels = zip(*test)
    print("Held-out days, rows actual background/ship, columns predicted background/ship")
    print("Calibrated threshold", threshold)
    print(confusion_matrix(labels, model.predict_proba(features)[:, 1] >= threshold))
    np.savez(HERE / "classifier.npz", coef=model.coef_[0], bias=model.intercept_[0], threshold=threshold)


if __name__ == "__main__":
    main()
