import os
# Open a file
import statistics
import math
import sys
import numpy as np


def main(query, train, qgram_size, user_k):
    qgram_tag = 'q_%d' % qgram_size
    k_tag = 'k_%d' % user_k
    result_path = './data/result/%s/%s/%s/' % (query, train, qgram_tag)
    truth_path = './data/truth/%s/%s/' % (query, train)

    stats_path = './data/stats/%s/%s/' % (query, train)
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    # list_of_result_files = (os.listdir(result_path))
    list_of_truth_files = (os.listdir(truth_path))

    # accuracy_directory = "C:/Users/Haonan/Desktop/accuracy/gps_20161002_trajectory/gps_20161001_trajectory/"
    scorelist = []
    totalScore = 0
    filesCounter = 0
    results = []
    for truth in list_of_truth_files:
        truthScore = 0
        resultScore = 0
        if ".txt" in truth:
            print(truth)
            fTruth = open(truth_path + truth, "r")
            fResult = open(result_path + truth, "r")
            truthLines = fTruth.readlines()
            resultLines = fResult.readlines()
            if len(resultLines) < user_k + 1 or len(truthLines) < user_k + 1:
                print('skipped')
                continue
            # because first line is ID ,so we start from second line, that's why i is in range (1 , k+1)
            filesCounter += 1
            topKResult = []
            topKTruth = []
            for i in range(1, user_k+1):
                tokens1 = truthLines[i].split(' ')
                # print(resultLines[i])
                # print(truth)
                tokens2 = resultLines[i].split(' ')
                # token1[0] is the ID, token1[1] is the distance
                topKTruth.append(tokens1[0])
                truthScore += float(tokens1[1])
                topKResult.append(tokens2[0])
                resultScore += float(tokens2[1])
            correct = len(set(topKResult).intersection(set(topKTruth)))
            accuracy = (float(correct)/user_k)*100
            scorelist.append(accuracy)
            # print(truthLines[0], accuracy)
            results.append((truth.replace("\n", ""), np.around(accuracy, 4)))
            fResult.close()
            fTruth.close()
        totalScore += math.pow((truthScore - resultScore)/user_k, 2)

    sorted_results = sorted(results, key=lambda tup: tup[1])
    sorted_results = [item[0] + ': ' + str(item[1]) for item in sorted_results]
    content = '\n'.join(sorted_results)
    finalScore = math.sqrt(totalScore) / filesCounter
    stats = "Total Accuracy: %s \nFinalScore : %s" % (str(statistics.mean(scorelist)), str(finalScore) + "\n")
    fAccuracy = open(stats_path + "%s_%s.txt" % (qgram_tag, k_tag), "w")
    fAccuracy.write(stats + content)
    fAccuracy.close()


if __name__ == '__main__':
    query_trajectory = sys.argv[1]
    train_trajectory = sys.argv[2]
    n = int(sys.argv[3])
    top_k = int(sys.argv[4])
    main(query=query_trajectory, train=train_trajectory, qgram_size=n, user_k=top_k)
