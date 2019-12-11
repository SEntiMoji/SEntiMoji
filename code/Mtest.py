"""
code for mcnemar"s test

Specify the method name and dataset name in the command line argument.
For example, if you want to do mcnemar"s test for the result of SEntiMoji and SEntiMoji-T
on Jira dataset, run:
python Mtest.py -methodA=SEntiMoji -methodB=SEntiMoji-T -dataset=Jira
"""
import argparse
import codecs
from statsmodels.sandbox.stats.runs import mcnemar


def load_method_res(method, task, dataset, emotion_type=None):
    correct_set = set()
    incorrect_set = set()
    idx = 0

    for fold in range(0, 5):
        
        if task == "sentiment":
            result_path = "../result/sentiment/%s/fold%d/result_%s.txt" % (dataset, fold, method)
        else:
            trans_dict = {"Jira" : "JIRA", "StackOverflow" : "SO"}
            assert(dataset in trans_dict and emotion_type is not None)
            result_path = "../result/emotion/%s/fold%d/result_%s_%s_%s.txt" % (dataset, fold, trans_dict[dataset], emotion_type.upper(), method)
        
        with codecs.open(result_path, "r", "utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                # predict correct
                if line[1] == line[2]:
                    correct_set.add(idx)

                # predict incorrect
                else:
                    incorrect_set.add(idx)
                idx += 1

    res = {"method": method, "dataset": dataset, "length": idx,
           "correct": correct_set, "incorrect": incorrect_set}
    return res


def mcnemar_test(methodA_res, methodB_res):
    assert methodA_res["dataset"] == methodB_res["dataset"]
    assert methodA_res["length"] == methodB_res["length"]

    # calculate the contingency table for mcnemar"s test
    #				       methodB
    #  	              correct  incorrect
    # methodA   correct    A        B
    #         incorrect    C        D

    A, B, C, D = 0, 0, 0, 0
    for idx in range(0, methodA_res["length"]):
        if idx in methodA_res["correct"] and idx in methodB_res["correct"]:
            A += 1
        if idx in methodA_res["correct"] and idx in methodB_res["incorrect"]:
            B += 1
        if idx in methodA_res["incorrect"] and idx in methodB_res["correct"]:
            C += 1
        if idx in methodA_res["incorrect"] and idx in methodB_res["incorrect"]:
            D += 1
    table = [[A, B], [C, D]]

    test_result = mcnemar(table, exact=False, correction=True)
    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methodA", type=str, required=True, choices=["SEntiMoji", "SEntiMoji-T", "SEntiMoji-G", "Senti4SD", "SentiCR", "SentiStrength", "SentiStrength-SE", "Down_EmoTxt", "EmoTxt"], help="name of method A")
    parser.add_argument("--methodB", type=str, required=True, choices=["SEntiMoji", "SEntiMoji-T", "SEntiMoji-G", "Senti4SD", "SentiCR", "SentiStrength", "SentiStrength-SE", "Down_EmoTxt", "EmoTxt"], help="name of method B")
    parser.add_argument("--dataset", type=str, required=True, choices=["Jira", "StackOverflow", "CodeReview", "JavaLib"], help="name of dataset")
    parser.add_argument("--task", type=str.lower, required=True, choices=["sentiment", "emotion"], help="specify task (emotion or sentiment)")
    parser.add_argument("--emotion_type", type=str.lower, required=False, choices=["love", "anger", "joy", "sad", "deva"], help="spcify emotion type for emotion detection task")
    args = parser.parse_args()
    
    methodA_res = load_method_res(args.methodA, args.task, args.dataset, args.emotion_type)
    methodB_res = load_method_res(args.methodB, args.task, args.dataset, args.emotion_type)
    
    test_result = mcnemar_test(methodA_res, methodB_res)
    print("%s vs. %s dataset = %s" % (args.methodA, args.methodB, args.dataset))
    print("mcnemar's test: statistic=%.3f, p-value=%.3f" % (test_result[0], test_result[1]))
