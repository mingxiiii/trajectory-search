from __future__ import print_function
import sys
# from pyspark.sql import SparkSession
from rtree.index import Rtree
from src.features.helper import *
from pyspark import SparkContext, SparkConf
import logging
import os
import gc


def main(query, train, query_num):
    logger = logging.getLogger('search_rtree')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('./log/%s' % query)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('Calculate common Q-grams for query trajectories')
    query_path = './data/processed/%s.txt' % query
    rtree_path = './data/interim/%s/my_rtree' % train

    logger.info('Query trajectory path: %s' % query_path)
    logger.info('Rtree path: %s' % rtree_path)

    query_data = load_trajectory(query_path, n=query_num)
    logger.info('Load %d query trajectories' % query_num)

    qry_qgram, qry_id_list = build_qgram(query_data)
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

    if not os.path.exists('./data/interim/%s' % query):
        os.mkdir('./data/interim/%s' % query)
    if not os.path.exists('./data/interim/%s/%s' % (query, train)):
        os.mkdir('./data/interim/%s/%s' % (query, train))

    candidate_traj_path = './data/interim/%s/%s/candidate_trajectory.txt' % (query, train)
    save_pickle(all_data, candidate_traj_path)
    logger.info('Output candidate_trajectory: %s' % candidate_traj_path)

    query_id_dict_path = './data/interim/%s/%s/query_id_dict.txt' % (query, train)
    logger.info('Output query_id_dict: %s' % query_id_dict_path)
    save_pickle(qry_id_dict, query_id_dict_path)
    gc.collect()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: query trajectory <file>, train trajectory <file>, query number <int>", file=sys.stderr)
        sys.exit(-1)
    query_trajectory = sys.argv[1]
    train_trajectory = sys.argv[2]
    n = int(sys.argv[3])
    main(query=query_trajectory, train=train_trajectory, query_num=n)



