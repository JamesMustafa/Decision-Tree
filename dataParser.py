import pandas
import time
from algorithm import trainTestSplit, buildDecisionTree, decisionTreePredictions, calculateAccuracy

dataFrame = pandas.read_csv("data/data_big_5.csv")
# dataFrame = dataFrame.drop("id", axis=1)
# dataFrame = dataFrame.drop("member_id", axis=1)
dataFrame = dataFrame[dataFrame.columns.tolist()[1:] + dataFrame.columns.tolist()[0: 1]]

termMapping = {"36 months": 1, "60 months": 2}
dataFrame["term"] = dataFrame["term"].map(termMapping)

gradeMapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
dataFrame["grade"] = dataFrame["grade"].map(gradeMapping)

homeOwnershipMapping = {"RENT": 1, "OWN": 2, "MORTGAGE": 3, "ANY": 4, "OTHER": 5, "NONE": 6}
dataFrame["home_ownership"] = dataFrame["home_ownership"].map(homeOwnershipMapping)

verificationStatusMapping = {"Not Verified": 1, "Source Verified": 2, "Verified": 3}
dataFrame["verification_status"] = dataFrame["verification_status"].map(verificationStatusMapping)

# loanStatusMapping = {"Charged Off": 1, "Current": 2, "Fully Paid": 3, "Late (31-120 days)": 4, "Default": 5}
# dataFrame["loan_status"] = dataFrame["loan_status"].map(loanStatusMapping)

# purposeMapping = {"credit_card": 1, "car": 2, "small_business": 3, "wedding": 4, "major_purchase": 5,
#                   "home_improvement": 6, "debt_consolidation": 7, "vacation": 8, "other": 9, "medical": 10,
#                   "house": 11, "moving": 12, "renewable_energy": 13}
# dataFrame["purpose"] = dataFrame["purpose"].map(purposeMapping)

dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize=0.65)

print("Decision tree algorithm starting...")

i = 1
accuracyTrain = 0
while accuracyTrain < 100:
    startTime = time.time()
    decisionTree = buildDecisionTree(dataFrameTrain, maxDepth=i)
    buildingTime = time.time() - startTime
    decisionTreeTestResults = decisionTreePredictions(dataFrameTest, decisionTree)
    accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
    decisionTreeTrainResults = decisionTreePredictions(dataFrameTrain, decisionTree)
    accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("maxDepth = {}: ".format(i), end="")
    print("accTest = {0:.2f}%, ".format(accuracyTest), end="")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end="")
    print("buildTime = {0:.2f}s".format(buildingTime), end="\n")
    i += 1
