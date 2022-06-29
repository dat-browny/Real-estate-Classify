import numpy as np
from keras.models import load_model
from encode.Encode import encoding
from metrics.Metrics import f1
from process_data.Process import pre_process

toLabel = {0: 'food_drink',
            1: 'entertainments',
            2: 'education',
            3: 'sport_fitness',
            4: 'civic_com',
            5: 'beauty',
            6: 'business',
            7: 'travel',
            8: 'science',
            9: 'parenting',
            10: 'recruiments',
            11: 'animals',
            12: 'vehicle',
            13: 'faith_spirituality',
            14: 'buy_sell',
            15: 'fashion',
            16: 'artworks',
            17: 'law_policy',
            18: 'health',
            19: 'relationship_identity',
            20: 'others',
            21: 'real_estate'}

def predict_label(data):
    path = '/Users/brownyeyes/Project/Real-estate-Classify/checkpoint/BiGRU.h5'
    model = load_model(path, custom_objects={"f1": f1})
    data = encoding(pre_process(data))
    pred = model.predict(data)
    label =  np.argmax(pred, axis=1)
    return toLabel[int(label)]