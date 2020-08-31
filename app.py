from flask import Flask, jsonify, request, render_template
import torch
import os
import sys
from transformers import CamembertTokenizer


sys.path.insert(1, r"D:\IsItCorrect\ML\Beta")
from custom_model import getCustomModel

# Get and initialize the tokenizer and some other useful stuffs
tokenizer_location = r"D:\Datasets\camembert-base\sentencepiece.bpe.model"
tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
softmax = torch.nn.Softmax(dim=1)
save_model_location = r"D:\Datasets\IsItCorrect\model\model_1.pth"

predict_correct_text = "Your sentence is Correct! :)"
predict_wrong_text = "Your sentence is Wrong! :("

app = Flask(__name__)


def load_model(save_model_location):
    model = getCustomModel()
    model.load_state_dict(torch.load(save_model_location))
    model.eval()
    return model


model = load_model(save_model_location)


def tokenize_sentence(sentence):
    sentence = tokenizer("sentence")
    sentence["input_ids"] = torch.tensor(sentence["input_ids"], dtype=torch.long)
    sentence["attention_mask"] = torch.tensor(sentence["attention_mask"])
    return sentence


def get_prediction(sentence):
    tokenized_sentence = tokenize_sentence(sentence)
    output = model(tokenized_sentence["input_ids"],
                   tokenized_sentence["attention_mask"])
    output = softmax(output)
    pred = torch.argmax(output).item()
    probab = output[pred].item() * 100.0
    return pred, probab

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        data = request.form.value()
        prediction, probability = get_prediction(data)
        if prediction:
            return render_template('index.html',
                                   prediction_text=predict_correct_text+" with a probability of: {}%".format(probability))
        else:
            return render_template('index.html',
                                   prediction_tex=predict_wrong_text+" with a probability of: {}%".format(probability))
