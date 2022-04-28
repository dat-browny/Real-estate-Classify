import numpy as np
def predict_label(model, data):
    pred = model.predict(data)
    label =  np.argmax(pred, axis=1)
    if label == 0:
        print("0 - Bài viết không thuộc chủ đề BĐS")
    elif label == 1:
        print("1 - Bài viết thuộc về chủ đề rao bán")
    elif label == 2:
        print("2 - Bài viết thuộc về chủ đề dự án")
    else:
        print("3 - Bài viết có nội dung về CĐT")