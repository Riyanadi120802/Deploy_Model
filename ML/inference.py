import tensorflow as tf
from model import Model
from preprocessing import Preprocessing

MODEL_PATH = "ML\models"
model = Model(model_path=MODEL_PATH)

def inference(input: base64):
    img_input = Preprocessing(input)
    result = model.predict(img_input) # tf
    result = model(img_input) # pytorch
    return result #[0.3 0.7]

def show_result():
    img = img_input
    class_img = inference()
    return img, class_img