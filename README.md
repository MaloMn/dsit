# **D**eep **S**peech **I**ntelligibility **T**oolkit

This work is based on Sondes Abderrazek's thesis.

## Running this program
### Creating the environment
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Getting intelligbility score, confusion matrix and ANPS scores
```bash
python main.py "file_stem" (--model "dsit/models/cnn" --action "all")
```
*Parameters in brackets already have the specified default values, so they can be omitted.*
> **_NOTE:_**  Audio (`.wav`) and phonetic transcription (`.lbl` or `.seg`) files should be put in the `dsit/audio/` folder, with the same name.
> For instance: 
> - `dsit/audio/patient_1.wav`
> - `dsit/audio/patient_1.lbl`  
> In this case, the file stem is "patient_1"

    parser.add_argument("-m", "--model", default="dsit/models/cnn")
    parser.add_argument("-a", "--action", default="all")