# MIT-BIH Arrhythmia Database

## Source
PhysioNet - MIT-BIH Arrhythmia Database
https://physionet.org/content/mitdb/1.0.0/

## Description
The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel 
ambulatory ECG recordings, obtained from 47 subjects studied by the BIH 
Arrhythmia Laboratory between 1975 and 1979.

## Sampling Information
- Sampling frequency: 360 Hz
- 11-bit resolution over a 10 mV range
- Two leads for each recording

## Usage
Use the `wfdb` library to read the ECG signals:
```python
import wfdb
record = wfdb.rdrecord('datasets/mit-bih/100')
annotation = wfdb.rdann('datasets/mit-bih/100', 'atr')
```

## Files
To download a subset of records, use:
```python
import wfdb
wfdb.dl_database('mitdb', 'datasets/mit-bih', records=['100', '101', '102'])
```

## Classes
Common arrhythmia types:
- N: Normal beat
- L: Left bundle branch block
- R: Right bundle branch block
- A: Atrial premature beat
- V: Premature ventricular contraction
