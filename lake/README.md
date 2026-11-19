# Run the lake listener

From the repository root, set up Python 3.12 once. Linux audio capture also needs PortAudio installed.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r lake/requirements.txt
```

Connect the hydrophone. List the audio inputs and replace 2 with its input number. Press Ctrl+C to stop.

```sh
python lake/listen.py --devices
python lake/listen.py --mic --device 2
```

To try a recording, run `python lake/listen.py --wav "/path/to/recording.wav"`. The number beside each result is a model score. The alert threshold is chosen using a separate sample of background recordings. Audio stays on the computer.

## Model

The bundled Google YAMNet sound model processes mono audio at 16 kHz. The listener removes the clip’s average offset and adjusts its level before scoring. A logistic classifier averages the 521 sound scores across each clip and learns to separate ship recordings from ambient ocean recordings. The training script samples up to 800 clips per class for training, 200 per class for threshold selection, and 200 per class for testing. Each split uses separate recording dates within each class. Each class comes from a different collection, so the classifier may also recognize recording equipment or site noise. [YAMNet model](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) · [Original model download](https://storage.googleapis.com/audioset/yamnet.tflite)

To refit the bundled classifier, run `python lake/train.py "/path/to/AWhispernetData"`. It expects the original `prop/oceanship_complete` and `ambient_ocean` folders with dated WAV filenames and writes `lake/classifier.npz`.
In the final archive check, EchoDot flagged 190 of 200 ship clips and raised false alarms on 41 of 200 background clips. The two collections were recorded separately. A test with observed boat passes and quiet periods at the same lake is still needed.
