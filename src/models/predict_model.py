from os import path
import sys
# sys.path.append(path.abspath('/Users/mingxidai/Documents/project/trajectory-search/'))
sys.path.append(path.abspath('/Users/mingxidai/Documents/Master/traj-dist-master'))

import pickle
import time
import traj_dist.distance as tdist
from src.features.build_bbox import load_trajectory
import numpy as np
#
# RDD: (test-key,Iter[(train_key,count),(train_key,count)])
# change threshold
threshold = 0.05
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
    # loading the files:
    with open("./data/processed/candidate_trajectory.txt", "rb") as f:
        candidateList = pickle.load(f)
    # candidateList => [[queryID_1,[(traID1, count1),(traID2, count2)]], [...]]
    with open("./data/processed/query_id_dict.txt", "rb") as f:
        query_id_dict = pickle.load(f)
    with open("./data/processed/rtree_id_dict2.txt", "rb") as f:
        rtree_id_dict = pickle.load(f)
    trajectory_dict = load_trajectory("./data/processed/gps_20161001_trajectory.txt")
    real_query_dict = load_trajectory("./data/processed/gps_20161002_query.txt")
    for index in range(len(candidateList)):
        # start to calculate:
        topK = candidateList[index][1][0:k]
        queryID = candidateList[index][0]
        # reverse the query Dict: fakeID -> realID
        query_id_dict = {v: k for k, v in query_id_dict.items()}
        # get the candidate trajectory IDs from top k
        pre_result = list(map(lambda x: x[0], topK))
        # build a map to save the result
        result_map = {}
        for t in pre_result:
             # result_map[t] = calculateEdr(trajectory_dict[rtree_id_dict[t]], real_query_dict[query_id_dict[queryID]])
            result_map[t] = tdist.edr(np.array(trajectory_dict[rtree_id_dict[t]]),np.array(real_query_dict[query_id_dict[queryID]]), "spherical")*max(len(trajectory_dict[rtree_id_dict[t]]),len(real_query_dict[query_id_dict[queryID]]))
        #print(result_map)
        fullCandidates = candidateList[index][1] # list of [ID, count]
        i = k
        query_tra = real_query_dict[query_id_dict[queryID]]
        lengthQ = len(query_tra)
        bestSoFar = result_map[topK[i-1][0]]
        while i < len(fullCandidates):
            candidate = fullCandidates[i]
            candidateID = candidate[0]
            try:
                tra_s = trajectory_dict[rtree_id_dict[candidateID]]
            # Mingxi: you can delete the "try except" part if you think there will no "miss match" in the dict anymore:
            except KeyError:
                pass
            countValue = candidate[1]
            lengthS = len(tra_s)
            if countValue >= (max(lengthQ,lengthS) - (bestSoFar+1)*qGramSize):
                # pointedByCounts = filter(lambda e:e[1]==countValue, fullCandidates)
                # for s in pointedByCounts:
                    realDist = tdist.edr(np.array(tra_s), np.array(query_tra), "spherical")*max(len(tra_s), len(query_tra))
                    if realDist < bestSoFar:
                        result_map[candidateID] = realDist
                        # update the best so far:
                        bestSoFar = sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[k-1][1]
            else:
                break
            i += 1

        finalResult = sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[0:k]
        with open("./data/result/query%s.txt" % index, 'w') as f:
            for item in finalResult:
                f.write("%s\n" % item)


if __name__ == "__main__":
    # param n means the number of results, (top n results)
    searchResult(7)
