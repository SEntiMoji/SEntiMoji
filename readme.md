
# README
This repository contains the data, code, pre-trained models and experiment results for the paper: **[SEntiMoji: An Emoji-Powered Learning Approach for Sentiment Analysis in Software Engineering]** (Anonymous authors for blind review).

## SEntiMoji
This study proposes SEntiMoji, which leverages the texts containing emoji from both Github and Twitter to improve the sentiment analysis task in software engineering (SE) domain. SEntiMoji is demonstrated to be able to significantly outperform the exisiting SE-customized sentiment analysis methods on representative benchmark datasets.

## Overview
* data/ contains the data used in this study. It contains two subfolders:
  - GitHub_data/ contains the processed emoji-texts used to train SEntiMoji.
  - benchmark_dataset/ contains the benchmark datasets used for evaluation, i.e., the JIRA, Stack Overflow, Code Review, and Java Library datset.

* code/ contains the scripts of SEntiMoji model. The variants of SEntiMoji share the same scripts with it. 
  - SEntiMoji_script/ contains the representation learning code (Deepmoji/deepmoji), the pipeline code for training and evaluating (pipeline.py), the files mapping labels to class indexes (label2index/), and vocabulary dicts for each pre-trained representation model (vocabulary/).
  - Mtest.py is responsible for the McNemar’s test.

* trained_model/ contains the pre-trained embeddings, representation models, and final sentiment classifier. It contains three subfolders:
  - word_embeddings/ contains the word embeddings trained on GitHub posts. 
  - representation_model/ contains the pre-trained representation models used for SEntiMoji (i.e., model_SEntiMoji.hdf5), SEntiMoji-G (i.e., model_SEntiMoji-G.hdf5), and SEntiMoji-T (i.e., model_SEntiMoji-T.hdf5). 

* result/ contains the detailed results of five-fold cross-validation (summarized in the sheets of result_5fold.xlsx) instead of the mean performance shown in the paper. In addition, for each dataset, we show the predicted labels for all folds. In each result file, the first column is the text, the second column is the predicted label, and the third column is the ground truth label.



## Running SentiMoji
1. We assume that you're using Python 3.6 with pip installed. As a backend you need to install either Theano (version 0.9+) or Tensorflow (version 1.3+). To run the code, you need the following dependencies:
 - [Keras](https://github.com/fchollet/keras) (above 2.0.0)
 - [scikit-learn](https://github.com/scikit-learn/scikit-learn)
 - [h5py](https://github.com/h5py/h5py)
 - [text-unidecode](https://github.com/kmike/text-unidecode)
 - [emoji](https://github.com/carpedm20/emoji)
 - [argparse](https://docs.python.org/3/library/argparse.html)
 - [codecs](https://docs.python.org/3/library/codecs.html)

If you lack some of the above dependencies, you can install it with pip.

2. In order to train a sentiment classifer based on SEntiMoji (or the variants of SEntiMoji) model, you can run the scripts in the code/SEntiMoji_scripts directory. 
For example, if you want to train and evaluate the classifier on the Jira dataset using the SEntiMoji representation model, navigate to code/SEntiMoji_scripts/ directory and run:
`python pipeline.py -model=SEntiMoji -dataset=Jira`

If you want to try another model or dataset, just change the argument of the command line. (`-model=[SEntiMoji, SEntiMoji-T, SEntiMoji-G], -dataset=[StackOverflow, Jira, CodeReview, JavaLib]`)

## Declaration
1. We upload all the benchmark datasets to this repository for convenience. As they were not generated and released by us, we do not claim any rights on them. If you use any of them, please make sure you fulfill the licenses that they were released with and consider citing the original papers. The scripts of baseline methods ([SentiStrength](http://sentistrength.wlv.ac.uk/), [SentiStrength-SE](http://laser.cs.uno.edu/resources/ProjectData/SentiStrength-SE_v1.5.zip), [SentiCR](https://github.com/senticr/SentiCR), [Senti4SD](https://github.com/collab-uniba/Senti4SD))  are not included in this repository. You can turn to their homepage for downloading.
2. The large-scale Tweets used to train DeepMoji are not released by [Felbo et al.](https://arxiv.org/abs/1708.00524) due to licensing restrictions. Therefore, we include the pre-trained DeepMoji released rather than the raw Tweet corpus in this repository.
3. The large-scale GitHub data are collected by [Lu et al.](https://arxiv.org/pdf/1812.04863.pdf) and not released publicly. After obtain their consent, in this repository, we release only the processed emoji-texts used to train our model, to increase reproducibility and replicability.

## License
This code and the pretrained model is licensed under the MIT license (https://mit-license.org).
