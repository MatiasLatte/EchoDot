# EchoDot

I’m Matias Latte. I built EchoDot to listen for boats through an underwater microphone. There are two ways to try it.

**1. WhisperNet · my original model.** I trained WhisperNet to recognize propeller sounds. The original demo runs on the sample recordings included here. [Try WhisperNet](processing/README.md)

**2. Lake listener · based on Google’s YAMNet.** Google trained the sound model. This version adds a classifier trained on ship and ocean recordings to flag possible boats. It can listen to a connected hydrophone or check a saved recording. [Try the lake listener](lake/README.md)

The listening range on a lake is still unknown. Measuring it takes boat passes at known distances and quiet periods to count false alarms.
