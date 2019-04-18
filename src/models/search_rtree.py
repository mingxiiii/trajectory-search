from __future__ import print_function
import sys
# from pyspark.sql import SparkSession
from rtree.index import Rtree
from src.features.build_bbox import *
from pyspark import SparkContext, SparkConf


def main(query_path, rtree_path):

    query = load_trajectory(query_path)
    qry_qgram, qry_id_list = build_qgram(query)
    qry_id_dict = build_id_dict(qry_id_list)  # key: query_id, value: query_key
    data_index = Rtree(rtree_path)

    conf = SparkConf().setAppName("PythonWordCount").setMaster("local")
    sc = SparkContext(conf=conf)

    all_data = []
    for qry_id, qry_qgrams in qry_qgram.items():
        qry_key = qry_id_dict[qry_id]
        data = []
        for qry_qgram in qry_qgrams:
            matches = [hit.object for hit in data_index.intersection(qry_qgram, objects=True)]
            matches = set(matches)
            data.append(list(matches))
        flat_data = [item for sublist in data for item in sublist]
        # print(flat_data)
        dist_data = sc.parallelize(flat_data)

        map_data = dist_data.map(lambda x: (x, 1))
        reduce_data = map_data.reduceByKey(lambda a, b: a+b).sortBy(lambda x: x[1], ascending=False).collect()
        # print(reduce_data)
        all_data.append([qry_key, reduce_data])
    save_pickle(all_data, './data/processed/candidate_trajectory.txt')
    save_pickle(qry_id_dict, './data/processed/query_id_dict.txt')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: query trajectory <file>, rtree <file>", file=sys.stderr)
        sys.exit(-1)
    trajectory = sys.argv[1]
    rtree = sys.argv[2]
    main(query_path=trajectory, rtree_path=rtree)



