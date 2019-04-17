from os import path
from src.features.build_bbox import *
import sys

sys.path.append(path.abspath('/Users/mingxidai/Documents/Master/traj-dist-master'))
import traj_dist.distance as tdist
import numpy as np


def main(query_path, train_path, query_id_path, train_id_path):
    train_data = load_trajectory(train_path)  # realID -> vectors
    query_data = load_trajectory(query_path)
    query_id_dict = read_pickle(query_id_path)  # fakeID -> readID
    train_id_dict = read_pickle(train_id_path)
    train_id_dict = {v: k for k, v in train_id_dict.items()}
    # print(train_id_dict)

    result = []
    for query_id, query_trajectory in query_data.items():
        query_key = query_id_dict[query_id]
        print(query_key)
        distance_list = []
        train_key_list = []
        for train_id, train_trajectory in train_data.items():
            try:
                train_key = train_id_dict[train_id]
                distance = tdist.edr(np.array(train_trajectory), np.array(query_trajectory), "spherical")*max(len(train_trajectory),len(query_trajectory))
                distance_list.append(distance)
                train_key_list.append(train_key)
            except KeyError:
                pass
        ix = sorted(range(len(distance_list)), key=lambda k: distance_list[k])
        distance_list_sorted = [distance_list[i] for i in ix]
        train_key_sorted = [train_key_list[i] for i in ix]
        trajectory_result = [(e1, e2) for e1, e2 in zip(train_key_sorted, distance_list_sorted)]
        result.append([query_key, trajectory_result])
    return result


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) != 5:
        # print("Usage: query trajectory <file>, train trajectory <file>, query id dictionary <file>, train id dictionary <file>", file=sys.stderr)
        sys.exit(-1)
    query = sys.argv[1]
    train = sys.argv[2]
    query_id = sys.argv[3]
    train_id = sys.argv[4]
    main(query_path=query, train_path=train, query_id_path=query_id, train_id_path=train_id)
