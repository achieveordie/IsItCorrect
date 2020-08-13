from transformers import CamembertConfig, CamembertModel, CamembertTokenizer


def getConfiguration():
    configuration = {
        "vocab_size": 32005,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
    }
    return configuration


def getTokenizer():
    tokenizer_location = r"D:\Datasets\camembert-base\sentencepiece.bpe.model"
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
    return tokenizer


def loadInitialModel():
    configuration = getConfiguration()
    config = CamembertConfig(vocab_size=configuration["vocab_size"],
                             hidden_size=configuration["hidden_size"],
                             num_hidden_layers=configuration["num_hidden_layers"],
                             intermediate_size=configuration["intermediate_size"],
                             hidden_act=configuration["hidden_act"],
                             hidden_dropout_prob=configuration["hidden_dropout_prob"],
                             attention_probs_dropout_prob=configuration["attention_probs_dropout_prob"],
                             max_position_embeddings=configuration["max_position_embeddings"])

    cam_model = CamembertModel(config).from_pretrained('camembert-base')
    return cam_model

