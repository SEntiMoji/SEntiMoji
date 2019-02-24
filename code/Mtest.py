'''
code for mcnemar's test

Specify the method name and dataset name in the command line argument.
For example, if you want to do mcnemar's test for the result of SEntiMoji and SEntiMoji-T
on Jira dataset, run:
python Mtest.py -methodA=SEntiMoji -methodB=SEntiMoji-T -dataset=Jira
'''

import codecs
from statsmodels.sandbox.stats.runs import mcnemar


def load_method_res(method, dataset):
    correct_set = set()
    incorrect_set = set()
    idx = 0

    for fold in range(0, 5):
        with codecs.open('../result/%s/fold%d/result_%s.txt' % (dataset, fold, method), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                # predict correct
                if line[1] == line[2]:
                    correct_set.add(idx)

                # predict incorrect
                else:
                    incorrect_set.add(idx)
                idx += 1

    res = {'method': method, 'dataset': dataset, 'length': idx,
           'correct': correct_set, 'incorrect': incorrect_set}
    return res


def mcnemar_test(methodA_res, methodB_res):
    assert methodA_res['dataset'] == methodB_res['dataset']
    assert methodA_res['length'] == methodB_res['length']

    # calculate the contingency table for mcnemar's test
    #				       methodB
    #  	              correct  incorrect
    # methodA   correct    A        B
    #         incorrect    C        D

    A, B, C, D = 0, 0, 0, 0
    for idx in range(0, methodA_res['length']):
        if idx in methodA_res['correct'] and idx in methodB_res['correct']:
            A += 1
        if idx in methodA_res['correct'] and idx in methodB_res['incorrect']:
            B += 1
        if idx in methodA_res['incorrect'] and idx in methodB_res['correct']:
            C += 1
        if idx in methodA_res['incorrect'] and idx in methodB_res['incorrect']:
            D += 1
    table = [[A, B], [C, D]]

    test_result = mcnemar(table, exact=False, correction=True)
    return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-methodA', type=str, default='SEntiMoji', help='name of method A')
    parser.add_argument('-methodB', type=str, default='SEntiMoji-T', help='name of method B')
    parser.add_argument('-dataset', type=str, default='Jira', help='name of dataset')
    args = parser.parse_args()
    
    # parse argument
    methodA = args.methodA
    methodB = args.methodB
    dataset = args.dataset
    
    methodA_res = load_method_res(methodA, dataset)
    methodB_res = load_method_res(methodB, dataset)
    
    test_result = mcnemar_test(methodA_res, methodB_res)
    print("%s vs. %s dataset = %s" % (methodA, methodB, dataset))
    print("mcnemar's test: statistic=%.3f, p-value=%.3f" % (test_result[0], test_result[1]))
