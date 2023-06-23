from preprocess.Text import Processer

p = Processer("./data/spacenews-2022.csv")
p.remove_small_words(user_savepoint=True)