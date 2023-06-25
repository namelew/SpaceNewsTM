import pandas as pd
from nltk.corpus import words,stopwords
from gensim.models.phrases import Phrases, Phraser
from geotext import GeoText
import spacy
import gensim
import string
import os


class Collection:
    def __init__(self, docs:list[list[str]] = None) -> None:
        self.docs:list[list[str]] = docs
        self.size:int = 0
        self.meanSize:int = 0
        self.nWords:int = 0
        self.greaterDoc:int = [0, 0]
        self.smallerDoc:int = [0, 0]
        if docs != None:
            self.evaluate()
        else:
            self.docs = []
    def evaluate(self):
        self.nWords = 0
        self.greaterDoc = self.smallerDoc = None
        self.size = 0
        for index,doc in enumerate(self.docs):
            lendoc = len(doc)
            if index == 0:
                self.greaterDoc = self.smallerDoc = [index, lendoc]
            self.greaterDoc = [index, lendoc] if lendoc > self.greaterDoc[1] else self.greaterDoc
            self.smallerDoc = [index, lendoc] if lendoc < self.smallerDoc[1] else self.smallerDoc
            self.nWords += lendoc
            self.size += 1

        print(f"Collection Size: {self.size}")
        print(f"Number of Words: {self.nWords}")
        print(f"Greater Doc: (ID: {self.greaterDoc[0]}, SIZE: {self.greaterDoc[1]})")
        print(f"Smaller Doc: (ID: {self.smallerDoc[0]}, SIZE: {self.smallerDoc[1]})")
        print(f"Mean Doc Size: {round(self.nWords/self.size)}")
class Processer:
    def __init__(self, filename:str) -> None:
        self.filename:str = filename
        self.outfile:str = filename[:filename.index('.') + 1]
        self.words = set(words.words())
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.collection:Collection = Collection()
    def __do_savepoint__(self, id:str):
        content = ""
        for doc in self.collection.docs:
            line = " ".join(doc)
            content += line + "\n"

        with open("./data/" + f"{id}-{self.outfile}.txt", "w") as file:
            file.write(content)
    def __load_savepoint__(self, id:str):
        with open("./data/" + f"{id}-{self.outfile}.txt", "r") as file:
            for line in file:
                self.collection.docs.append((line.replace("\n", "")).split(" "))
        self.collection.evaluate()
    def __clean_garbage(self):
        def clean(doc: str) -> str:
            stop_free = " ".join(w for w in doc.lower().split() if w not in self.stop_words and len(w) < 15)
            punc_free = ''.join(' ' if w in self.punctuation else w for w in stop_free)
            t_words = " ".join(w for w in punc_free.lower().split() if w in self.words)
            return t_words

        self.collection.docs = [clean(d) for d in self.collection.docs]
        self.collection.evaluate()
    def __lemmatization(self):
        lemmatizer = spacy.load('en_core_web_lg')
        for i,d in enumerate(self.collection.docs):
            d=lemmatizer(d) # transform the documento into a spacy object
            d=[w.lemma_ for w in d] #recrite the doc with the lemmas
            self.collection.docs[i]=d #update document d at position i in collection
        self.collection.evaluate()
        self.__do_savepoint__("lemma")
    def __grams_builder(self, use_savepoint=False, savepoint="lemma"):
        if use_savepoint:
            self.__load_savepoint__(savepoint)

        phrases = Phrases(self.collection.docs, min_count=4, threshold=.5, scoring='npmi')
        bigram = Phraser(phrases)
        self.collection.docs = [bigram[doc] for doc in self.collection.docs]

        phrases = Phrases(self.collection.docs, min_count=2, threshold=.7, scoring='npmi')
        trigram = Phraser(phrases)
        self.collection.docs = [trigram[doc] for doc in self.collection.docs]

        bigm = {word for doc in self.collection.docs for word in doc if '_' in word}
        print(f"Number of N-Grams: {len(bigm)}")
        self.collection.evaluate()
        self.__do_savepoint__("n-grams")
    def __remove_small_words(self,use_savepoint=False, savepoint="n-grams"):
        if use_savepoint:
            self.__load_savepoint__(savepoint)
        self.collection.docs = [gensim.utils.simple_preprocess(" ".join(doc), deacc= True, min_len=3) for doc in self.collection.docs]
        self.collection.evaluate()
        self.__do_savepoint__("smallfree")
    def from_csv(self, columns:list[str]):
        df = pd.read_csv(self.filename)

        for row in df.iterrows():
            try:
                doc = ' '.join(row[1][column] for column in columns)
                geo = GeoText(doc)
                self.words.update([city.lower() for city in geo.cities])
                self.words.update([country.lower() for country in geo.countries])
                self.words.update([nationality.lower() for nationality in geo.nationalities])
                self.collection.docs.append(doc)
            except:
                continue

        self.collection.evaluate()
    def from_txt(self):
        with open(self.filename, "r") as file:
            for line in file:
                self.collection.docs.append((line.replace("\n", "")).split(" "))
                self.collection.evaluate()
    def transform(self, use_savepoint=False, savepoint="smallfree") -> list[list[str]]:
        if use_savepoint and os.path.isfile(f"./data/{savepoint}-{self.outfile}.txt"):
            self.__load_savepoint__(savepoint)
            self.__do_savepoint__("processed")
            return self.collection.docs
    
        print("Clean docs")
        self.__clean_garbage()
        print("Lemmatizing words")
        if not use_savepoint or not os.path.isfile(f"./data/lemma-{self.outfile}.txt"):
            self.__lemmatization()
        print("Buid N-Grams")
        if not os.path.isfile(f"./data/n-grams-{self.outfile}.txt"):
            self.__grams_builder(use_savepoint=use_savepoint)
        print("Remove small words")
        self.__remove_small_words()
        self.__do_savepoint__("processed")

        return self.collection.docs