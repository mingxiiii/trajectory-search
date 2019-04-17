from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
from rtree.index import Rtree

from src.features.build_bbox import *
from sklearn import preprocessing

from pyspark import SparkContext, SparkConf


def main(query_path, rtree_path, rtree_id_dict):

    query = load_trajectory(query_path, n=1)
    qry_qgram, qry_id_list = build_qgram(query)
    qry_id_dict = build_id_dict(qry_id_list)

    params = {
        'qgrams': qry_qgram,
        'rtree_path': rtree_path,
        'qry_dict': qry_id_dict,
        'rtree_id_dict': rtree_id_dict
    }
    data = get_matches(**params)

    conf = SparkConf().setAppName("PythonWordCount").setMaster("local")
    sc = SparkContext(conf=conf)

    distData = sc.parallelize(data, n=4)
    # distData.foreach(print)
    # eps = 0.2
    # subtraction = distData.map(lambda x_y: (x_y[0], tuple(np.abs(np.subtract(x_y[0], x_y[1]))))).filter(lambda x_y: x_y[0][0]<eps and x_y[0][1]<eps)
    # pairs = subtraction.map(lambda x_y: (x_y[0], 1))
    # counts = pairs.reduceByKey(lambda a, b: a + b).sortBy(False)
    # counts_pairs = counts.map(lambda x_y: (x_y[0][0], (x_y[0][1], x_y[1])))
    # counts_list = counts_pairs.reduceByKey(lambda a, b: a + b)
    #
    # counts_list.saveAsTextFile('../data/interim/candidates')


def get_matches(**kwargs):
    data = [] # init list to store key-value pairs
    qgrams = kwargs['qgrams']  # query qgrams
    qry_dict = kwargs['qry_dict']  # query key-id mapping

    rtree_path = kwargs['rtree_path']
    data_index = Rtree(rtree_path)  # load rtree
    print(data_index)
    rtree_dict = read_id_dict(kwargs['rtree_id_dict'])  # load tree_dict mapping
    counter = 0
    for qry_id, qry_qgrams in qgrams.items():
        qry_key = qry_dict[qry_id]
        # print('order id', qry_id)
        # print('qgrams', qry_qgrams)
        # print('query key', qry_key)
        for qry_qgram in qry_qgrams:
            matches = [hit.object for hit in data_index.intersection(qry_qgram, objects=True)]
            matches = set(matches)
            print(matches)
            break

    #         for hit in hits:
    #             tree_key = hit.id
    #             order_id = hit.object[0]
    #             hit_qgrams = hit.object[1]
    #             tree_dict[order_id] = tree_key
    #             for hit_qgram in hit_qgrams:
    #                 item = ((qry_key, tree_key), (np.subtract(qry_qgram, hit_qgram)))
    #                 counter += 1
    #                 data.append(item)
    #                 print(item)
    #             print(tree_key)
    #             print(obj)
    #             break
    # print(counter)
    # save_id_dict(tree_dict, './data/interim/rtree_dict')
    return data


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: query trajectory <file>, rtree <file>, rtree dictionary <file>", file=sys.stderr)
        sys.exit(-1)
    trajectory = sys.argv[1]
    rtree = sys.argv[2]
    rtree_dict = sys.argv[3]
    main(query_path=trajectory, rtree_path=rtree, rtree_id_dict=rtree_dict)



