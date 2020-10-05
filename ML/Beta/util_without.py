from transformers import CamembertConfig, CamembertModel


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
        "max_position_embeddings": 514,
        "type_vocab_size": 1
    }
    return configuration


def loadInitialModel():
    configuration = getConfiguration()
    config = CamembertConfig(vocab_size=configuration["vocab_size"],
                             hidden_size=configuration["hidden_size"],
                             num_hidden_layers=configuration["num_hidden_layers"],
                             intermediate_size=configuration["intermediate_size"],
                             hidden_act=configuration["hidden_act"],
                             hidden_dropout_prob=configuration["hidden_dropout_prob"],
                             attention_probs_dropout_prob=configuration["attention_probs_dropout_prob"],
                             max_position_embeddings=configuration["max_position_embeddings"],
                             type_vocab_size=configuration["type_vocab_size"])

    cam_model = CamembertModel(config) #.from_pretrained('./model')
    return cam_model
