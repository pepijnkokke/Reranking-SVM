Ranking translations
====================

We will be usinga linear classifier to rank translations that came out of a phrase based translation system.

Other assignments
------

* Find our [IBM Models repository here](https://github.com/pepijnkokke/UvA-MT1-IBM)
* Find our [Phrase-Based MT repository here](https://github.com/pepijnkokke/UvA-MT2-PBT)

Getting started
------

First expand all files in `data`.

To setup the development and running environment execute the following commands:

```bash
# make sure you are using a recent pip/virtualenv version
python -m pip install -U pip virtualenv

# setup and start a virtual environment
virtualenv .env
source .env/bin/activate

# install dependancies in the virtual environment
pip install spacy
pip install jupyter
pip install nltk
pip install scipy
pip install sklearn
pip install msgpack-python

# download language model data
python -m spacy.en.download
python -m spacy.de.download
```
