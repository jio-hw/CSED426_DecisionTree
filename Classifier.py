# By Yuoa
# TODO: Integrate numerializing phases to one function (I couldn't do this due to the time limit)
# NOTE: For now, in Classifier.test, col lists after removing excludedCols must be same with trained col lists.
# NOTE: For now, comma in double quote is recognized as splitter. Please preprocess them before run.

import os, copy, graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

class Classifier:
    def __init__(self, csvPath, hasHeader=False, answerCol=-1, exclude=[], numeralDistance=10, featureWeight=[], dtcConfig={}, fitConfig={}):
        # Initialize variables
        self.nominalData = {}
        self.nominalAnswer = []
        self.data = []
        self.answer = []
        self.answerHeader = None
        self.fitConfig = fitConfig
        csvLines = []

        # Open CSV and get all lines first
        csv = open(csvPath, 'r')
        csvRawLines = csv.readlines()
        csv.close()

        # Split each lines
        for rl in csvRawLines:
            csvLines.append(rl.strip().split(','))

        # Exclude given attributes
        exclusions = []
        if hasHeader:
            for ex in exclude:
                if ex not in csvLines[0]:
                    print("Cannot found given exclusions.")
                    exit(1)
                exclusions.append(csvLines[0].index(ex))
        exclusions.sort(reverse=True)
        for line in csvLines:
            for ex in exclusions:
                line.pop(ex)

        # Set header column
        if hasHeader:
            hdr = csvLines.pop(0)
            if type(answerCol) is str:
                if answerCol not in hdr:
                    print("Cannot found given answer column.")
                    exit(1)
                self.answerHeader = answerCol
                answerCol = hdr.index(answerCol)
        else:
            if answerCol == -1:
                print("Use the last element as answer column.")
                answerCol = len(csvLines[0]) - 1

        # Prepare for converting nominal data
        for i in range(0, len(csvLines[0])):
            self.nominalData[i] = []
        
        # Prepare data
        for line in csvLines:
            self.answer.append(line.pop(answerCol))
            self.data.append(line)

        # Covert nominal categories to integer
        rowNumber = 0
        for row in self.data:
            colNumber = 0
            if len(csvLines) > 100000 and rowNumber % 10000 == 0:
                print(str(rowNumber) + "th data is now going to be processed...")
            for col in row:
                colData = 0
                try:
                    colData = float(col) # Numeral
                except ValueError:
                    if col not in self.nominalData[colNumber]:
                        colData = float(len(self.nominalData[colNumber])) * numeralDistance
                        self.nominalData[colNumber].append(col)
                    else:
                        colData = float(self.nominalData[colNumber].index(
                            col)) * numeralDistance
                self.data[rowNumber][colNumber] = colData
                colNumber += 1
            rowNumber += 1

        # Convert nominal answers to float
        ansNumber = 0 
        for ans in self.answer:
            ansData = 0
            try:
                ansData = float(ans)
            except ValueError:
                if ans not in self.nominalAnswer:
                    ansData = len(self.nominalAnswer)
                    self.nominalAnswer.append(ans)
                else:
                    ansData = float(self.nominalAnswer.index(ans))
            self.answer[ansNumber] = ansData
            ansNumber += 1

        # Fitting models
        self.classifier = DecisionTreeClassifier(**dtcConfig)
        self.classifier.fit(self.data, self.answer, **fitConfig)

    def numeralize(self, row):
        # Convert nominal categories to integer
        colNumber = 0
        for col in row:
            colData = 0
            try:
                colData = float(col)
            except ValueError:
                if col not in self.nominalData[colNumber]:
                    print("Unknown argument on column " + str(colNumber) + ": " + str(col))
                    colData = len(self.nominalData[colNumber])
                else:
                    colData = float(self.nominalData[colNumber].index(col))
            row[colNumber] = colData
            colNumber += 1
        return row

    def classify(self, rows):
        # Check if it is single row
        if type(rows[0]) is not list:
            rows = [rows]

        # Classify
        resultData = []
        rawResult = self.classifier.predict(rows)
        for result in rawResult:
            if len(self.nominalAnswer) > 0:
                resultData.append(self.nominalAnswer[int(result)])
            else:
                resultData.append(result)

        return resultData

    def probability(self, rows):
        # Check if it is single row
        if type(rows[0]) is not list:
            rows = [rows]

        # Get probability
        return self.classifier.predict_proba(rows)

    def test(self, csvPath, hasHeader=False, excludeInPrediction=[], includeInOutput=[], answerType=float, exportHeader=False, exportPath="./out.csv", exportFullPredictions=False):
        csvRawLines = []
        csvLines = []
        hdr = []
        excludedCols = []
        includedCols = []

        # Open CSV and get all lines first
        csv = open(csvPath, 'r')
        csvRawLines = csv.readlines()
        csv.close()

        # Split each lines with numeralizing
        for rl in csvRawLines:
            csvLines.append(rl.strip().split(','))

        # Get exclusion columns
        if hasHeader:
            hdr = csvLines.pop(0)
            for ex in excludeInPrediction:
                if ex not in hdr:
                    print("Unknown exclusion header: " + ex)
                    return []
                excludedCols.append(hdr.index(ex))
            for ic in includeInOutput:
                if ic[0] not in hdr:
                    print("Unknown exclusion header: " + ic)
                    return []
                includedCols.append([hdr.index(ic[0]), ic[1]])
        else:
            hdr = range(0, len(csvLines[0]))
            excludedCols = excludeInPrediction
            for i in hdr:
                includedCols.append([i, float])
        excludedCols.sort(reverse=True)
        
        # Generate prediction dataset
        predictionInput = []
        copiedLines = copy.deepcopy(csvLines)
        for data in copiedLines:
            tempData = data
            for ec in excludedCols:
                tempData.pop(ec)
            #print(tempData)
            predictionInput.append(self.numeralize(tempData))
        
        # Predict
        prediction = []
        if exportFullPredictions:
            prediction = self.probability(predictionInput)
        else:
            prediction = self.classify(predictionInput)

        # Prepare output data
        outData = []
        
        if exportHeader:
            outHeader = []
            for ic in includedCols:
                outHeader.append(hdr[ic[0]])
            if exportFullPredictions:
                outHeader += self.nominalAnswer
            else:
                outHeader.append(self.answerHeader)
            outData.append(outHeader)

        itera = 0
        for p in prediction:
            outLine = []
            for ic in includedCols:
                outLine.append(str(ic[1](csvLines[itera][ic[0]])))
            if exportFullPredictions:
                predLine = []
                for r in p:
                    predLine.append(str(r))
                outLine += predLine
            else:
                outLine.append(str(answerType(p)))
            outData.append(outLine)
            itera += 1

        # Make output
        os.makedirs(os.path.dirname(exportPath), exist_ok=True)
        out = open(exportPath, 'w')
        for data in outData:
            out.write(",".join(data) + "\n")
        out.close()

    def fit(self):
        self.classifier.fit(self.data, self.answer, **self.fitConfig)

    def export(self, name, type='png'):
        graphviz.Source(export_graphviz(self.classifier)).render(name, format=type)

def diff(fileA, fileB):
    # Read and get lines
    a = open(fileA, 'r')
    b = open(fileB, 'r')
    al = a.readlines()
    bl = b.readlines()
    a.close()
    b.close()

    # Compare
    size = len(al)
    same = 0
    for i in range(size):
        if al[i] == bl[i]:
            same += 1

    # Return
    return same / size
