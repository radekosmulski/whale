# Humpback Whale Identification Competition Starter Pack

The code in this repo is all you need to make a first submission to the [Humpback Whale Identification Competition](https://www.kaggle.com/c/humpback-whale-identification). It uses the [FastAi library](https://github.com/fastai/fastai) release 1.0.36.post1 (this is important - you are likely to encounter an error if you use any other version of the library). 

For additional information please refer to the discussion thread on [Kaggle forums](https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/69409).

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
1. first_submission - getting all the basics in place
2. new_whale detector - binary classifer known_whale / new_whale
3. only_known
