# **D**eep **S**peech **I**ntelligibility **T**oolkit

This work is based on Sondes Abderrazek's thesis.

## Running this program
This program was developed and tested under Python 3.10.

### Creating the environment
You can either create a virtualenv...
```bash
virtualenv .venv
source .venv/bin/activate
```
...or a conda environment.
```bash
conda create --name dsit python=3.10
conda activate dsit
```

You can then install required Python packages.
```bash
pip install -r requirements.txt
```

### Getting intelligibility and severity scores, confusion matrix and ANPS scores
```bash
python main.py path/to/audio.wav path/to/transcription.lbl (--model "dsit/models/cnn" --action "all")
```
*Parameters in brackets already have the specified default values, so they can be omitted.*
> **_NOTE:_**  Both `.lbl` and `.seg` files are supported for aligned transcriptions.
> When providing a `.seg` file, it will automatically be parsed as a `.lbl` file in the same folder as the `.seg` file.
