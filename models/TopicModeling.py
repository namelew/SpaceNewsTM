import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
import os

class TopicModeler:
    def __init__(self, collection:list[list[str]] = None) -> None:
        self.collection:list[list[str]] = []
        if collection == None:
            with open("./data/" + "processed-spacenews-2022.txt", "r") as file:
                for line in file:
                    self.collection.append((line.replace("\n", "")).split(" "))
        else:
            self.collection:list[list[str]] = collection
        self.bagOfWords:list[list[tuple[int,int]]] = None
        self.bagOfDocs:pd.DataFrame = None
    def BuildBags(self, use_savepoint=False):
        self.__build_bag_of_words()
        self.__build_bag_of_docs(use_savepoint=use_savepoint)
    def __build_bag_of_words(self):
        dictionary = Dictionary(self.collection)
        dictionary.filter_extremes(keep_n = 50000, no_above= 0.2, no_below = 3)
        self.bagOfWords = [dictionary.doc2bow(d) for d in self.collection]
    def __build_bag_of_docs(self, use_savepoint=False):
        if use_savepoint and os.path.isfile("./data/bag-docs-spacenews-2022.csv"):
            self.bagOfDocs = pd.read_csv("./data/bag-docs-spacenews-2022.csv")
            return
        uniquewords=set()
        for d in self.collection:
            [uniquewords.add(w) for w in d]

        coldict=[]
        for d in self.collection:
            coldict.append(dict.fromkeys(uniquewords,0))

        for i,d in enumerate(self.collection):
            for w in d:
                coldict[i][w]+=1
        columns=list(uniquewords)
        self.bagOfDocs=pd.DataFrame(coldict,columns=columns)
        self.bagOfDocs.to_csv("./data/bag-docs-spacenews-2022.csv", sep=',', index=False)
    def byClusters(self) -> None:
        # using kmeans and DBSCAN to eval the distribution of topics
        X = PCA(n_components=round(len(self.bagOfDocs) * 1)).fit_transform(self.bagOfDocs)

        model = cluster.KMeans(n_clusters=9)

        # Fit the model to the data
        model.fit(X)

        # Plot the data points and color them according to their cluster assignment
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
        plt.show()

        try:
            print(f"Model\nS-Score:{silhouette_score(X, model.labels_)}")
        except:
            print("Error evaluating model")