from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: wordcount <file>", file=sys.stderr)
    #     sys.exit(-1)
    #
    # spark = SparkSession\
    #     .builder\
    #     .appName("PythonWordCount")\
    #     .getOrCreate()
    # lines = spark.read.text('./data/processed/gps_20161002_query.txt').rdd.map(lambda r: r[0])
    # counts = lines.flatMap(lambda x: x.split(' ')) \
    #               .map(lambda x: (x, 1)) \
    #               .reduceByKey(add)
    # output = counts.collect()
    # for (word, count) in output:
    #     print("%s: %i" % (word, count))
    #
    # spark.stop()
    #
    conf = SparkConf().setAppName("PythonWordCount").setMaster("cs5425")
    sc = SparkContext(conf=conf)

    data = [((1, 2), ((1, 2,), (3, 4,))), ((1, 2), ((1, 2,), (3, 4,))), ((1, 2), ((1, 2,), (3, 4,))),
            ((1, 2), ((1, 2,), (3, 4,)))]
    distData = sc.parallelize(data)