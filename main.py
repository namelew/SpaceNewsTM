from preprocess.Text import Processer
from models.TopicModeling import TopicModeler

tm = TopicModeler()
tm.BuildBags(use_savepoint=True)
tm.byClusters()