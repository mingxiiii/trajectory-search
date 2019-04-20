import os
# Open a file
k = 50
truth_directory = "C:/Users/Haonan/Desktop/truth/gps_20161002_trajectory/gps_20161001_trajectory/"
result_directory = "C:/Users/Haonan/Desktop/result/gps_20161002_trajectory/gps_20161001_trajectory/"
list_of_truth_files = (os.listdir(truth_directory))
list_of_result_files = (os.listdir(result_directory))
accuracy_directory = "C:/Users/Haonan/Desktop/accuracy/gps_20161002_trajectory/gps_20161001_trajectory/"
fAccuracy = open(accuracy_directory + "top %s accuracy.txt" % k, "w")
for truth in list_of_truth_files:
    if ".txt" in truth:
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
            tokens2 = resultLines[i].split(' ')
            topKTruth.append(tokens1[0])
            topKResult.append(tokens2[0])
        correct = len(set(topKResult).intersection(set(topKTruth)))
        accuracy = (float(correct)/k)*100
        # print(truthLines[0], accuracy)
        fAccuracy.write(truth.replace("\n", "")+" : " + str(accuracy)+"\n")
        fResult.close()
        fTruth.close()

fAccuracy.close()




# Close opend file
