# By Yuoa

import sys
from Classifier import Classifier

if __name__ == "__main__":

    # Do Bank Prediction
    print("\nPrediction: Bank")
    bank = Classifier("data/Bank/bank_train.csv",
                          hasHeader=True, answerCol='y', numeralDistance=1, dtcConfig={"max_depth": 6, "max_leaf_nodes": None})
    sys.stdout.write("Saving results... ")
    bank.test("data/Bank/bank_test.csv", hasHeader=True, excludeInPrediction=['id'], includeInOutput=[
        ['id', int]], answerType=int, exportHeader=True, exportPath="./out/bank_result.csv")
    bank.export("out/bank_result", 'pdf')
    print('ok')

    # Do Crime Prediction
    print("\nPrediction: Crime")
    crime = Classifier("data/Crime/crime_train_mutant.csv",
                          hasHeader=True, answerCol='Category', numeralDistance=1, exclude=['Dates', 'Descript', 'Address', 'Resolution', 'MONTH'], dtcConfig={"max_depth": 12, "max_leaf_nodes": None})
    sys.stdout.write("Saving results... ")
    crime.test("data/Crime/crime_test_mutant.csv", hasHeader=True, excludeInPrediction=['id', 'Dates', 'Address', 'MONTH'], includeInOutput=[
        ['id', float]], exportHeader=True, exportPath="./out/crime_result.csv", exportFullPredictions=True)
    crime.export("out/crime_result", 'svg')
    print('ok')
