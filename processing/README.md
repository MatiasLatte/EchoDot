# WhisperNet

Matias Latte trained the original EchoDot model to flag propeller sounds. Its trained weights are included in [WhisperNet_model.h5](WhisperNet_model.h5). The demo picks a bundled recording, makes a working copy, and prints its prediction. A positive result saves that copy and its audio features in the detections folder.

From the repository root, use a Python 3.12 environment for the original model.

```sh
python3.12 -m venv .venv-whispernet
source .venv-whispernet/bin/activate
python -m pip install -r processing/requirements.txt
python main.py
```

Press Ctrl+C to stop. To use a live underwater microphone, choose the [YAMNet lake listener](../lake/README.md).
