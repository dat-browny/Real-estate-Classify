from keras.models import load_model
from encode.Encode import encoding
from metrics.Metrics import Metrics, f1
from process_data.Process import pre_process
from predict.Predict import predict_label 

if __name__ == "__main__":
    path = '/Users/brownyeyes/Project/Real-estate-Classify/checkpoint/model_BiLSTM.h5' #Path to model BiLSTM
    model = load_model(path, custom_objects={"f1": f1})
    text_input = input("Nhập bài viết: \n")
    text_vector = encoding(pre_process(text_input))
    predict_label(model, text_vector)
