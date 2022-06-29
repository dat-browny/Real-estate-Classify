import os
import sys
import json 
from keras.utils.data_utils import pad_sequences

root_encode = sys.path[0] + '/encode/'
dictionary = json.loads(open(os.path.join(root_encode,'dict.txt')).read())

def encoding(text):
  result = []
  for i in text.split():
    try:
      result.append(dictionary[i])
    except:
      result.append(0)
  return pad_sequences([result], maxlen=300, dtype='float32', padding='post', truncating='post')
