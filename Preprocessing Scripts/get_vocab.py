import time
from gensim.models import KeyedVectors

def load_embeddings():
    t_loadembed_1 = time.time()
    model_location = r'D:\Datasets\IsItCorrect\fra_word_embedding_100\model.bin'
    model = KeyedVectors.load_word2vec_format(model_location, binary=True)
    vocab = list(model.vocab.keys())
    vocab_save = r'vocab.txt'
    with open(vocab_save, 'w', encoding='utf8') as file:
        for vo in vocab:
            file.write(vo+" ")
    t_loadembed_2 = time.time()
    print("For load_embeddings, time is ", (t_loadembed_2-t_loadembed_1))


if __name__ == '__main__':
    load_embeddings()