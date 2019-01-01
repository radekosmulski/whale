# Humpback Whale Identification Competition Starter Pack

The code in this repo is all you need to make a first submission to the [Humpback Whale Identification Competition](https://www.kaggle.com/c/humpback-whale-identification). It uses the [FastAi library](https://github.com/fastai/fastai) release 1.0.36.post1 for anything up to point 7 in the *Navigating through the repository* list below (this is important - you are likely to encounter an error if you use any other version of the library). Subsequently I switch to 1.0.39.

For additional information please refer to discussion threads on Kaggle forums: [classification](https://www.kaggle.com/c/humpback-whale-identification/discussion/74647), [feature learning](https://www.kaggle.com/c/humpback-whale-identification/discussion/75984), [detection](...).

**Some people [reported issues](https://github.com/radekosmulski/whale/issues/1) with running the first_submission notebook. If you encounter the issue, you should be okay to skip to the subsequent notebooks. The one that scores 0.760 on the LB is `only_known_train.ipynb`.**

## Making first submission
1. Install the [fastai library](https://github.com/fastai/fastai), specifically version 1.0.36.post1. The easiest way to do it is to follow the developer install as outlined in the README of the fastai repository. Once you perform the installation, navigate to the fastai directory and execute `git checkout 1.0.36.post1`. You can verify that this worked by executing the following inside jupyter notebook or a Python REPL:
```
import fastai
fastai.__version__
```
2. Clone this repository. cd into data. Download competition data by running `kaggle competitions download -c humpback-whale-identification`. You might need to agree to competition rules on competition website if you get a 403.
3. Create the train directory and extract files via running `mkdir train && unzip train.zip -d train`
4. Do the same for test: `mkdir test && unzip test.zip -d test`
5. Open `first_submission.ipynb` in jupyter notebook and run all cells.

## Navigating through the repository

Here is the order in which I worked on the notebooks:
1. [first_submission](https://github.com/radekosmulski/whale/blob/master/first_submission.ipynb) - getting all the basics in place
2. [new_whale_detector](https://github.com/radekosmulski/whale/blob/master/new_whale_detector.ipynb) - binary classifer known_whale / new_whale
3. [oversample](https://github.com/radekosmulski/whale/blob/master/oversample.ipynb) - addressing class imbalance
4. [only_known_research](https://github.com/radekosmulski/whale/blob/master/only_known_research.ipynb) - how to modify the architecture and what hyperparams to use
5. [only_known_train](https://github.com/radekosmulski/whale/blob/master/only_known_train.ipynb) - training on full dataset
6. [resize](https://github.com/radekosmulski/whale/blob/master/resize.ipynb) - resize the images before training to free up CPU
7. [siamese network](https://github.com/radekosmulski/whale/blob/master/siamese_network_prototype.ipynb) - a fully working prototype of a siamese network
8. **!!! Important !!!** - to make use of some of the new functionality available in fast.ai at this point I switch to 1.0.39.
9. [fluke detection](https://github.com/radekosmulski/whale/blob/master/fluke_detection.ipynb) - train a model to draw bounding boxes surrounding flukes
10. [fluke detection redux](https://github.com/radekosmulski/whale/blob/master/fluke_detection_redux.ipynb) - better results, less code, works with current fastai master
