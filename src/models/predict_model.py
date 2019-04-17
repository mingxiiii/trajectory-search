from pyspark import SparkContext, SparkConf
import sparkpickle

conf = SparkConf().setAppName("PythonWordCount").setMaster("local")
sc = SparkContext(conf=conf)


# RDD: (test-key,Iter[(train_key,count),(train_key,count)])
# change threshold
threshold = 0.5

def match(coor1,coor2):
    return abs(coor1[0]-coor2[0]) <= threshold and abs(coor1[1]-coor2[1]) <= threshold


def calculateEdr(trajectory1, trajectory2):
    if len(trajectory1) == 0:
        return len(trajectory2)
    elif len(trajectory2) == 0:
        return len(trajectory1)
    else:
        return min(calculateEdr(trajectory1[1:], trajectory2[1:]) + subcost(trajectory1[0], trajectory2[0]),
                   calculateEdr(trajectory1[1:], trajectory2)+1,
                   calculateEdr(trajectory1, trajectory2[1:])+1)


def subcost(t1, t2):
    if match(t1, t2):
        return 1
    else:
        return 0


def searchResult(k):
    with open("/path/to/file", "rb") as f:
        inputRdd = sparkpickle.load(f)
    topK = inputRdd.flatMap(lambda x: x[1]).take(k)
    trainkey1 = topK[0][0] # trainkey
    count0 = topK[0][1] # count

