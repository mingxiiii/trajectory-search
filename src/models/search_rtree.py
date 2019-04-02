from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
from rtree.index import Rtree

from src.features.build_bbox import *
from sklearn import preprocessing

from pyspark import SparkContext, SparkConf


def main(query_path, rtree_path):

    query = load_trajectory(query_path, n=1)
    qry_qgram, qry_bbox, qry_id_list = build_qgram(query)
    qry_id_dict = build_id_dict(qry_id_list)
    params = {
        'qgrams': qry_qgram,
        'bbox': qry_bbox,
        'rtree_path': rtree_path,
        'qry_dict': qry_id_dict,
    }
    data = get_matches(**params)

    conf = SparkConf().setAppName("PythonWordCount").setMaster("cs5425")
    sc = SparkContext(conf=conf)

    distData = sc.parallelize(data)
    eps = 0.2
    subtraction = distData.map(lambda x_y: (x_y[0], tuple(np.abs(np.subtract(x_y[0], x_y[1]))))).filter(lambda x_y: x_y[0][0]<eps and x_y[0][1]<eps)
    pairs = subtraction.map(lambda x_y: (x_y[0], 1))
    counts = pairs.reduceByKey(lambda a, b: a + b)
    

def get_matches(**kwargs):
    data = [] # init list to store key-value pairs
    qgrams = kwargs['qgrams']  # query qgrams
    bbox = kwargs['bbox']  # query bounding box
    qry_dict = kwargs['qry_dict']  # query key-id mapping

    rtree_path = kwargs['rtree_path']
    data_index = Rtree(rtree_path)  # load rtree
    # tree_dict = read_id_dict()  # load tree_dict mapping
    tree_dict = {}
    for key, qry_qgrams in qgrams.items():
        qry_key = qry_dict[key]
        print('orderid', key)
        print('qgrams', qry_qgrams)
        print('bounding box', bbox[key])
        print('query key', qry_key)
        hits = data_index.intersection(bbox[key], objects=True)
        for qry_qgram in qry_qgrams:
            for hit in hits:
                tree_key = hit.id
                order_id = hit.object[0]
                hit_qgrams = hit.object[1]
                tree_dict[order_id] = tree_key
                for hit_qgram in hit_qgrams:
                    item = ((qry_key, tree_key), (qry_qgram, hit_qgram))
                    data.append(item)
                # print(tree_key)
                # print(obj)
                # break
    save_id_dict(tree_dict, './data/interim/rtree_dict')
    return data


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: query trajectory <file>, rtree <file>", file=sys.stderr)
        sys.exit(-1)
    trajectory = sys.argv[1]
    rtree = sys.argv[2]
    main(query_path=trajectory, rtree_path=rtree)



