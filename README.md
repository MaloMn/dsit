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

`example.json` contains an example of an output file.
It can be used to adapt downstream tasks to the output architecture.
It contains dummy data.

Below is an overview of this output file.
This file contains:
- both intelligibility and severity scores ;
- the phonemes confusion matrix ;
- local ANPS scores per phonetic trait.
```json
{
    "intelligibility": {
        "value": 4.29,
        "minimum": 0,
        "maximum": 10
    },
    "severity": {
        "value": 3.22,
        "minimum": 0,
        "maximum": 10
    },
   "confusion_matrix": {
      "labels": ["sil", "a", ..., "z", "\u0292"],
       "matrix": [[...], ..., [...]]
    },
   "anps": {
      "vowel": {
        "+nasal": 0.9,
        "-nasal": 0.75,
        ...
      },
      "consonant": {
        "+sonorant": 0.9,
        "-sonorant": 0.75,
        ...
      }
   }
}
```
