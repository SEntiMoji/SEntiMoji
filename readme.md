# ReadMe
This repository contains the data, code, pre-trained models and experiment results for the paper: **[SEntiMoji: An Emoji-Powered Learning Approach for Sentiment Analysis in Software Engineering]** .

## SEntiMoji
This study proposes SEntiMoji, which leverages the texts containing emoji from both Github and Twitter to improve the sentiment analysis and emotion detection task in software engineering (SE) domain. SEntiMoji is demonstrated to be able to significantly outperform the exisiting SE-customized sentiment analysis and emotion detection methods on representative benchmark datasets.

## Overview
* data/ contains the data used in this study. It contains two subfolders:
  - GitHub_data/ contains the processed emoji-texts used to train SEntiMoji.
  - benchmark_dataset/ contains the benchmark datasets used for evaluation. Benchmark dataset includes datasets for sentiment analysis task and emotion detection task.
    + Datasets for sentiment analysis: [the Jira](http://ansymore.uantwerpen.be/system/files/uploads/artefacts/alessandro/MSR16/archive3.zip.), [Stack Overflow](https://github.com/collab-uniba/Senti4SD), [Code Review](https://github.com/senticr/SentiCR/), and [Java Library datset](https://sentiment-se.github.io/replication.zip).
    + Datasets for emotion detection: [the Jira Emotion Dataset (for binary classification)](http://ansymore.uantwerpen.be/system/files/uploads/artefacts/alessandro/MSR16/archive3.zip), [the Jira Deva Dataset (for multiclass classification)](https://figshare.com/s/277026f0686f7685b79e), [the Stack Overflow Emotion Dataset (for binary classification)]( https://github.com/collab-uniba/EmotionDatasetMSR18).

* code/ contains the scripts of SEntiMoji model. The variants of SEntiMoji share the same scripts with it. 
  - SEntiMoji_script/ contains the representation learning code (Deepmoji/deepmoji), the pipeline code for training and evaluating (pipeline.py), the files mapping labels to class indexes (label2index/), and vocabulary dicts for each pre-trained representation model (vocabulary/).
  - Mtest.py is responsible for the McNemar’s test.

* trained_model/ contains the pre-trained embeddings, representation models, and final sentiment classifier. It contains three subfolders:
  - word_embeddings/ contains the word embeddings trained on GitHub posts. 
  - representation_model/ contains the pre-trained representation models used for SEntiMoji (i.e., model_SEntiMoji.hdf5), SEntiMoji-G (i.e., model_SEntiMoji-G.hdf5), and SEntiMoji-T (i.e., model_SEntiMoji-T.hdf5). 

  ⚠️ Since the size of model and embedding exceeds the Github file size limit, we use [git lfs](https://git-lfs.github.com/) to manage these large files. If you use ```git clone``` to download the whole project, these large files are not included so you will get error when you load them. You have to download them through one of these two following ways:  
  1. Install [git lfs](https://git-lfs.github.com/) first and use command ```git lfs pull``` to download the large files.
  2. Open the file in github website and click the download button to download the large files directly. 
  

* result/ contains the detailed results of five-fold cross-validation (summarized in the sheets of result_5fold_sentiment.xlsx and result_5fold_emotion.xlsx) instead of the mean performance shown in the paper. In addition, for each dataset, we show the predicted labels for all folds. In each result file, the first column is the text, the second column is the predicted label, and the third column is the ground truth label.


## Running SEntiMoji
1. We assume that you're using Python 3.6 with pip installed. As a backend you need to install either Theano (version 0.9+) or Tensorflow (version 1.3+). For the installation of depedencies, open the command line and run: 
`pip install -r requirements.txt`

2. In order to train a sentiment classifer or emotion detector based on SEntiMoji (or the variants of SEntiMoji) model, you can run the scripts in the code/SEntiMoji_script directory. 

 - For sentiment classification task, you have to specify the model, task, dataset in command line. For example, if you want to train and evaluate the classifier on the Jira dataset using the SEntiMoji representation model, navigate to code/SEntiMoji_scripts/ directory and run:
`python pipeline.py --model SEntiMoji --task sentiment --dataset Jira`.

 - For emotion detection task, you have to specify the model, task, dataset, emotion type in command line. For example, if you want to train and evaluate the classifier on the Jira LOVE dataset using the SEntiMoji representation model, navigate to code/SEntiMoji_scripts/ directory and run:
`python pipeline.py --model SEntiMoji --task emotion --dataset Jira --emotion_type love`.

If you want to try another model or dataset, just change the arguments of the command line. Use command `python pipeline.py --help` to see the detailed decriptions for command line arguments.

3. If you want to perform McNemar’s Test to compare the results of two classifiers, you can run Mtest.py in code/ directory. You have to specify the method name, dataset name and task name in the command line argument. 
 - For sentiment classification task: For example, if you want to do mcnemar's test for the result of SEntiMoji and SEntiMoji-T
on Jira dataset, run: `python Mtest.py --methodA SEntiMoji --methodB SEntiMoji-T --dataset Jira --task sentiment`.
 - For emotion detection task: For example, if you want to do mcnemar's test for the result of SEntiMoji and SEntiMoji-T
on Jira LOVE dataset, run: `python Mtest.py --methodA SEntiMoji --methodB SEntiMoji-T --dataset Jira --task emotion --emotion_type love`.

If you want to try another model or dataset, just change the arguments of the command line. Use command `python Mtest.py --help` to see the detailed decriptions for command line arguments.

## Declaration
1. We upload all the benchmark datasets to this repository for convenience. As they were not generated and released by us, we do not claim any rights on them. If you use any of them, please make sure you fulfill the licenses that they were released with and consider citing the original papers. The scripts of baseline methods ([SentiStrength](http://sentistrength.wlv.ac.uk/), [SentiStrength-SE](http://laser.cs.uno.edu/resources/ProjectData/SentiStrength-SE_v1.5.zip), [SentiCR](https://github.com/senticr/SentiCR), [Senti4SD](https://github.com/collab-uniba/Senti4SD), [EmoTxt](https://github.com/collab-uniba/Emotion_and_Polarity_SO), [DEVA](https://figshare.com/s/277026f0686f7685b79e))  are not included in this repository. You can turn to their homepage for downloading.

2. The large-scale Tweets used to train DeepMoji are not released by [Felbo et al.](https://arxiv.org/abs/1708.00524) due to licensing restrictions. Therefore, we include the pre-trained DeepMoji released rather than the raw Tweet corpus in this repository.

3. The large-scale GitHub data are collected by [Lu et al.](https://arxiv.org/pdf/1812.04863.pdf) and not released publicly. After obtain their consent, in this repository, we release only the processed emoji-texts used to train our model, to increase reproducibility and replicability.

## License
This code and the pretrained model is licensed under the MIT license (https://mit-license.org).

## Citation
Please consider citing the following paper when using our code or pretrained models for your application.
```
@inproceedings{chencao2019,
  title={SEntiMoji: An Emoji-Powered Learning Approach for Sentiment Analysis in Software Engineering},
  author={Zhenpeng Chen and Yanbin Cao and Xuan Lu and Qiaozhu Mei and Xuanzhe Liu},
  booktitle={Proceedings of the 2019 ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE'19},
  year={2019}
}
```
