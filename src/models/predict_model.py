from pyspark import SparkContext, SparkConf
import sparkpickle
import pickle
from src.features.build_bbox import load_trajectory
#
# RDD: (test-key,Iter[(train_key,count),(train_key,count)])
# change threshold
threshold = 0.5
qGramSize = 20

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
    with open("C:/Users/Haonan/PycharmProjects/trajectory-search/candidate_trajectory.txt", "rb") as f:
        candidateList = pickle.load(f)
    topK = candidateList[1][0:k]
    queryID = candidateList[0]
    print(topK)
    with open("C:/Users/Haonan/PycharmProjects/trajectory-search/rtree_id_dict.txt", "rb") as f:
        rtree_id_dict = pickle.load(f)
    with open("C:/Users/Haonan/PycharmProjects/trajectory-search/query_id_dict.txt", "rb") as f:
        query_id_dict = pickle.load(f)
    trajectory_dict = load_trajectory("C:/Users/Haonan/PycharmProjects/trajectory-search/gps_20161001_trajectory.txt")
    real_query_dict = load_trajectory("C:/Users/Haonan/PycharmProjects/trajectory-search/gps_20161002_query.txt")
    query_id_dict = {v: k for k, v in query_id_dict.items()}
    result = list(map(lambda x: x[0], topK))
    print(query_id_dict[queryID])
    print(result)


    result_map = {}
    for t in result:
         result_map[t] = calculateEdr(trajectory_dict[rtree_id_dict[t]], real_query_dict[query_id_dict[queryID]])
    print(result_map)


    fullCandidates = candidateList[1] # list of [ID, count]
    i = k
    query_tra = real_query_dict[query_id_dict[queryID]]
    lengthQ = len(query_tra)
    bestSoFar = result_map[topK[i][0]]
    while i < len(fullCandidates):
        candidate = fullCandidates[i]
        tra_s = trajectory_dict[rtree_id_dict[candidate[0]]]
        countValue = candidate[1]
        lengthS = len(tra_s)
        if countValue >= (max(lengthQ,lengthS) - (bestSoFar+1)*qGramSize):
            pointedByCounts = filter(lambda e:e[1]==countValue, fullCandidates)
            for s in pointedByCounts:
                realDist = calculateEdr(trajectory_dict[rtree_id_dict[s[0]]],query_tra)
                if(realDist<bestSoFar):
                    result_map[s[0]] = realDist
                    bestSoFar = sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[k-1]
        else:
            break
        i+=1

    return sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[0:k]



if __name__ == "__main__":
    print(calculateEdr([[1,2],[2,3],[3,4]],[[1,1],[2,2],[3,3],[4,4]]))
