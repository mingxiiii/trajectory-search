from rtree.index import Rtree
from src.features.build_bbox import *
import sys


def main(trajectory_path):
    trajectory = load_trajectory(trajectory_path)
    qgram, bbox, order_id_list = build_qgram(trajectory)
    order_id_dict = build_id_dict(order_id_list)

    # R-tree constructor
    # parameter: 'data_full' is the filename of R-tree storage
    #            2 files are created: data_full.dat, data_full.idx
    # return: r-tree index
    data_idx = Rtree('../data/processed/my_rtree')
    # put all trajectories into r-tree in the form of bounding box
    for key, box in bbox.items():
        # paremeters:
        #    1. node id: integer mapped from order_id
        #    2. bounding box: (x_min,y_min,x_max,y_max)
        #    3. data inside each node: array of (x,y) coordinates for all timestamps, timestamp info not saved
        data_idx.insert(order_id_dict[key], box, obj=(key, qgram[key]))
    print("Finished...")


if __name__ == '__main__':
    dtrain = sys.argv[1]
    main(dtrain)
