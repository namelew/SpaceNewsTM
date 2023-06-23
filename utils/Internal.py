import spacy
import nltk
from os import _exit

def dowloadDicts() -> None:
    try:
        spacy.cli.download("en_core_web_lg")
        nltk.download('stopwords')
        nltk.download('words')
    except Exception as err:
        print("Unable to download, internal dependencys from packages. Cause: ", err)
        _exit(1)

def Build():
    dowloadDicts()

Build()