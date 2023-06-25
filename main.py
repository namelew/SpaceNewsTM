from preprocess.Text import Processer

p = Processer("./data/spacenews-2022.csv")
p.from_csv(['title', 'content', 'author'])
p.transform(use_savepoint=True)