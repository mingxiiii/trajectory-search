from rtree.index import Rtree
from src.features.build_bbox import *

data_index = Rtree('./data/processed/my_rtree')

for object in enumerate(data_index):
    print(object)