# Real-estate-Classify

## Installation
```bash
pip install -r requirements.txt
```

## Install VnCoreNLP
```bash
mkdir -p vncorenlp/models/wordsegmenter
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```
## Guide to run code
In file Process.py, edit rdrsegmenter_path to VNCoreNLP jar file in folder above.
```python
rdrsegmenter_path = '...'
#Path to VnCoreNLP jar file
```
In file main.py, path to model_BiLSTM.h5 file.
```python
path = '/...' 
#Path to model BiLSTM
```
Then, run file main.py to predict.
