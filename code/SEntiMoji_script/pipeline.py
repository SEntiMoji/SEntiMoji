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

sys.path.append("DeepMoji/deepmoji/")

from sentence_tokenizer import SentenceTokenizer
from model_def import deepmoji_architecture, load_specific_weights
from finetuning import load_benchmark, finetune

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

MAX_LEN = 150


def load_data(filename):
    f = codecs.open(filename, "r", "utf-8")
    data_pair = []
    for line in f:
        line = line.strip().split("\t")
        data_pair.append((line[0], line[1]))
    return data_pair


def prepare_5fold(data_pair):
    sind = 0
    eind = 0
    random.shuffle(data_pair)
    fold_size = int(len(data_pair) / 5)
    for fold in range(0, 5):
        sind = eind
        eind = sind + fold_size
        train_pair = data_pair[0:sind] + data_pair[eind:len(data_pair)]
        test_pair = data_pair[sind:eind]
        yield (train_pair, test_pair)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["SEntiMoji", "SEntiMoji-T", "SEntiMoji-G"], help="name of pretrained representation model")
    parser.add_argument("--task", type=str.lower, required=True, choices=["sentiment", "emotion"], help="specify task (sentiment or emotion)")
    parser.add_argument("--benchmark_dataset_name", type=str, required=False, choices=["Jira", "StackOverflow", "CodeReview", "JavaLib"], help="name of benchmark dataset")
    parser.add_argument("--emotion_type", type=str.lower, required=False, default=None, choices=["anger", "love", "deva", "joy", "sad"], help="specify emotion dataset")
    
    parser.add_argument("--use_own_dataset", action='store_true', help="whether use your own dataset or not")
    parser.add_argument("--own_dataset_dir", type=str, required=False, default=None, help="directory of your train data file")
    parser.add_argument("--own_dataset_file", type=str, required=False, default=None, help="file name of your train data file")
    
    args = parser.parse_args()

    print("args:")
    d = args.__dict__
    for key,value in d.items():
        print("%s = %s"%(key,value))

    # parse arguments
    model_path = "../../model/representation_model/model_%s.hdf5" % args.model
    vocab_path = "vocabulary/vocabulary_%s.json" % args.model

    # load vocabulary 
    with open(vocab_path, "r") as f_vocab:
        vocabulary = json.load(f_vocab)

    try:
        # use provided dataset
        if not args.use_own_dataset:
            if args.benchmark_dataset_name is None:
                raise ValueError("should provide benchmark dataset name")

            if args.task == "sentiment":
                data_path = "../../data/benchmark_dataset/sentiment/%s.txt" % args.benchmark_dataset_name 
                label2index_path = "label2index/sentiment/label2index_%s.json" % args.benchmark_dataset_name 

            else:
                trans_dict = {"Jira" : "JIRA", "StackOverflow" : "SO"}
                if args.benchmark_dataset_name not in trans_dict:
                    raise ValueError("invalid dataset name for emotion task")
                data_file_name = "%s_%s" % (trans_dict[args.benchmark_dataset_name ], args.emotion_type.upper())
                data_path = "../../data/benchmark_dataset/emotion/%s/%s.txt" % (args.benchmark_dataset_name , data_file_name)

                if args.emotion_type == 'deva':
                    if args.benchmark_dataset_name != "Jira":
                        raise ValueError("invalide dataset name for deva, requires Jira")
                    label2index_path = "label2index/emotion/label2index_5class.json" 
                else:
                    label2index_path = "label2index/emotion/label2index_2class.json"
                
            # load data and label2index file
            data_pair = load_data(data_path)

            with open(label2index_path, "r") as f_label:
                label2index = json.load(f_label)
            index2label = {i: l for l, i in label2index.items()}
        
        # prepare your own data
        else:
            if args.own_dataset_dir is None or args.own_dataset_file is None:
                raise ValueError("should specify your own dataset directory and filename")

            # load data
            data_path = "{}/{}".format(args.own_dataset_dir, args.own_dataset_file)
            data_pair = load_data(data_path)

            # generate label2index file
            labels = set([pair[1] for pair in data_pair])
            label2index = {}
            for label in labels:
                label2index[label] = len(label2index)
            index2label = {i: l for l, i in label2index.items()}

            label2index_path = "{}/{}".format(args.own_dataset_dir, "label2index.json")
            with open(label2index_path, 'w') as f:
                json.dump(label2index, f)
    
    except RuntimeError as e:
        print("Error:", repr(e))

    # split 5 fold
    data_5fold = prepare_5fold(data_pair)

    # sentence tokenizer (MAXLEN means the max length of input text)
    st = SentenceTokenizer(vocabulary, MAX_LEN)
    fold = 0

    # 5 fold
    for item in data_5fold:
        # prepare training, validation, testing set
        train_text = [p[0] for p in item[0]]
        train_label = [p[1] for p in item[0]]
        test_text = [p[0] for p in item[1]]
        test_label = [p[1] for p in item[1]]

        train_X, _, _ = st.tokenize_sentences(train_text)
        test_X, _, _ = st.tokenize_sentences(test_text)
        train_y = np.array([label2index[l] for l in train_label])
        test_y = np.array([label2index[l] for l in test_label])

        nb_classes = len(label2index)
        nb_tokens = len(vocabulary)

        # use 20% of the training set for validation
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,
                                                          test_size=0.2, random_state=0)
        # model 
        model = deepmoji_architecture(nb_classes=nb_classes,
                                      nb_tokens=nb_tokens,
                                      maxlen=MAX_LEN, embed_dropout_rate=0.25, final_dropout_rate=0.5, embed_l2=1E-6)
        model.summary()

        # load pretrained representation model
        load_specific_weights(model, model_path, nb_tokens, MAX_LEN,
                              exclude_names=["softmax"])
        
        # train model
        model, acc = finetune(model, [train_X, val_X, test_X], [train_y, val_y, test_y], nb_classes, 100,
                              method="chain-thaw", verbose=2)
        
        pred_y_prob = model.predict(test_X)

        if nb_classes == 2:
            pred_y = [0 if p < 0.5 else 1 for p in pred_y_prob]
        else:
            pred_y = np.argmax(pred_y_prob, axis=1)

        # evaluation
        print("*****************************************")
        print("Fold %d" % fold)
        accuracy = accuracy_score(test_y, pred_y)
        print("Accuracy: %.3f" % accuracy)

        precision = precision_score(test_y, pred_y, average=None)
        recall = recall_score(test_y, pred_y, average=None)
        f1score = f1_score(test_y, pred_y, average=None)
        for index in range(0, nb_classes):
            print("label: %s" % index2label[index])
            print("Precision: %.3f, Recall: %.3f, F1 score: %.3f" % (precision[index], recall[index], f1score[index]))
        print("*****************************************")

        # save predict result
        if not args.use_own_dataset:
            if args.task == "sentiment":
                save_name = "result_%s_%s_fold%d.txt" % (args.model, args.benchmark_dataset_name, fold)
            elif args.task == "emotion":
                save_name = "result_%s_%s_%s_fold%d.txt" % (args.model, args.benchmark_dataset_name, args.emotion_type, fold)
        else:
            save_name = "result_fold%d.txt" % fold

        with open(save_name, "w") as f:
            for i in range(0, len(test_text)):
                f.write("%s\t%s\t%s\r\n" % (test_text[i], index2label[pred_y[i]], test_label[i]))

        fold += 1