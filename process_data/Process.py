import re
import numpy as np
from vncorenlp import VnCoreNLP
from pyvi import ViUtils

rdrsegmenter_path = '...'
#Path to VnCoreNLP jar file

annotator = VnCoreNLP(rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')
stopwords =['comments', 'recent', 'all', 'da ban', 'top', 'most', 'all', 'comment', 'like', 'share']

def remove_links(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'bit.ly/\S+', '', text)
    text = text.strip('[link]')
    return text 

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_special_characters(text):
    RE_HTML_TAG = re.compile(r'<[^>]+>')
    DATETIME = '\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}'
    text = re.sub(r'\n', '', text)
    text = re.sub(RE_HTML_TAG, '', text)
    text = re.sub(DATETIME, '', text)
    return text

def remove_stopword(text):
    return " ".join([word for word in str(text).split() if word not in stopwords]) 

def remove_number(text):
    return " ".join([word for word in str(text).split() if word.isalpha()])

def pre_process(text):
      text = remove_links(text)
      text = remove_punctuation(text)
      text = remove_special_characters(text)
      text = remove_number(text)
      try:
        text = ViUtils.add_accents(text)
      except: 
        pass 
      text = text.lower()
      text = remove_stopword(text)
      text = " ".join([" ".join(i) for i in annotator.tokenize(text)])
      return text

def to_tensor(x):
  return np.asarray(x).astype('float32')

def pre_process(text):
    text = remove_links(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = remove_number(text)
    try:
        text = ViUtils.add_accents(text)
    except: 
        pass 
    text = text.lower()
    text = remove_stopword(text)
    text = " ".join([" ".join(i) for i in annotator.tokenize(text)])
    return text