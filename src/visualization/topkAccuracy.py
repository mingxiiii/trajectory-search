import os
# Open a file
import statistics
import math
k =50

truth_directory = "C:/Users/Haonan/Desktop/truth/gps_20161002_trajectory/gps_20161001_trajectory/"
result_directory = "C:/Users/Haonan/Desktop/result/gps_20161002_trajectory/gps_20161001_trajectory/"
list_of_truth_files = (os.listdir(truth_directory))
list_of_result_files = (os.listdir(result_directory))
accuracy_directory = "C:/Users/Haonan/Desktop/accuracy/gps_20161002_trajectory/gps_20161001_trajectory/"
fAccuracy = open(accuracy_directory + "top %s accuracy_q25.txt" % k, "w")
scorelist = []
totalScore = 0
filesCounter = 0
for truth in list_of_truth_files:
    truthScore = 0
    resultScore = 0
    if ".txt" in truth:
        filesCounter += 1
        fTruth = open(truth_directory + truth, "r")
        fResult = open(result_directory + truth, "r")
        truthLines = fTruth.readlines()
        resultLines = fResult.readlines()
        # because first line is ID ,so we start from second line, that's why i is in range (1 , k+1)
        correct = 0
        assert len(truthLines) > k
        topKResult = []
        topKTruth = []
        for i in range(1, k+1):
            tokens1 = truthLines[i].split(' ')
            print(resultLines[i])
            print(truth)
            tokens2 = resultLines[i].split(' ')
            # token1[0] is the ID, token1[1] is the distance
            topKTruth.append(tokens1[0])
            truthScore += float(tokens1[1])
            topKResult.append(tokens2[0])
            resultScore += float(tokens2[1])
        correct = len(set(topKResult).intersection(set(topKTruth)))
        accuracy = (float(correct)/k)*100
        scorelist.append(accuracy)
        # print(truthLines[0], accuracy)
        fAccuracy.write(truth.replace("\n", "")+" : " + str(accuracy)+"\n")
        fResult.close()
        fTruth.close()
    totalScore += math.pow((truthScore - resultScore)/k, 2)

finalScore = math.sqrt(totalScore)/filesCounter
fAccuracy.write("Total Accuracy: " + str(statistics.mean(scorelist))+"\n")
fAccuracy.write("FinalScore : " + str(finalScore)+"\n")


fAccuracy.close()




# Close opend file
