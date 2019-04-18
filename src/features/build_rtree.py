import numpy as np
from rtree.index import Rtree
from src.features.build_bbox import *
import sys



def main(trajectory_path):
    data = load_trajectory(trajectory_path)
    trajectory, order_id_list = build_qgram(data)
    id_dict = build_id_dict(order_id_list)

    # R-tree constructor
    # parameter: 'data_full' is the filename of R-tree storage
    #            2 files are created: data_full.dat, data_full.idx
    # return: r-tree index
    data_idx = Rtree('../data/processed/my_rtree')
    # put all trajectories into r-tree in the form of bounding box
    node_id = 0
    for key,qgrams in trajectory.items():
        for qgram in qgrams:
        # paremeters:
        #    1. node id
        #    2. bounding box(point): (x,y,x,y)
        #    3. data inside each node: trajectory's key from order_dict
            data_idx.insert(node_id, (qgram[0],qgram[1],qgram[0],qgram[1]), obj=(id_dict[key]))
            node_id += 1
    print("Finished...")


if __name__ == '__main__':
    dtrain = sys.argv[1]
    main(dtrain)
