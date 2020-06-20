"""
Pipeline code for training and evaluating the sentiment classifier.
We use the Deepmoji architecture here, see https://github.com/bfelbo/DeepMoji for detail.
"""
import re
import codecs
import random
import numpy as np
import sys
import json
import argparse
import traceback

sys.path.append("DeepMoji/deepmoji/")

from sentence_tokenizer import SentenceTokenizer
from model_def import deepmoji_architecture, load_specific_weights

MAX_LEN = 150


def load_data(filename):
    f = codecs.open(filename, "r", "utf-8")
    data_pair = []
    for line in f:
        line = line.strip().split("\t")
        data_pair.append(line[0])
    return data_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="path of model weights")
    parser.add_argument("--test_file_path", type=str, required=True, help="path of test file")
    parser.add_argument("--pretrained_model", type=str, required=True, choices=["SEntiMoji", "SEntiMoji-T", "SEntiMoji-G"], help="name of representation model")
    parser.add_argument("--nb_classes", type=int, required=True, help="number of classification classes")
    parser.add_argument("--save_result_path", type=str, required=False, default="classification_results.txt", help="path of classification result")
    args = parser.parse_args()

    print("args:")
    d = args.__dict__
    for key,value in d.items():
        print("%s = %s"%(key,value))

    # parse arguments
    model_path = args.model_path
    test_path = args.test_file_path
    vocab_path = "vocabulary/vocabulary_%s.json" % args.pretrained_model
    nb_classes = args.nb_classes
    save_path = args.save_result_path

    # load vocabulary 
    with open(vocab_path, "r") as f_vocab:
        vocabulary = json.load(f_vocab)
    nb_tokens = len(vocabulary)

    test_text = load_data(test_path)

    # sentence tokenizer (MAXLEN means the max length of input text)
    st = SentenceTokenizer(vocabulary, MAX_LEN)

    # tokenize test text
    test_X, _, _ = st.tokenize_sentences(test_text)

    # load model 
    model = deepmoji_architecture(nb_classes=nb_classes,
                                      nb_tokens=nb_tokens,
                                      maxlen=MAX_LEN)

    load_specific_weights(model, model_path, nb_tokens, MAX_LEN, nb_classes=nb_classes)

    pred_y_prob = model.predict(test_X)

    if nb_classes == 2:
        pred_y = [0 if p < 0.5 else 1 for p in pred_y_prob]
    else:
        pred_y = np.argmax(pred_y_prob, axis=1)
    
    with open(save_path, "w") as f:
        for i in range(0, len(test_text)):
            f.write("{}\t{}\r\n".format(test_text[i], pred_y[i]))
    
    print("Results were saved to {}".format(save_path))
