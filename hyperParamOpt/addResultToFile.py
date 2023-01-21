import glob
import pandas as pd
import re
import sys

df = pd.read_csv("hyperParOptResults.csv")

for fileName in glob.glob('./jobsOutput/*.o*'):
    with open(fileName, "r") as file:
        print(fileName)
        hypePars = ''
        time = ''
        rocAUC = ''
        fOne = ''
        accuracy = ''
        
        lines = file.readlines()
        # Check if the last line starts with "Accuracy"
        if lines[-1].startswith("Accuracy"):
            for line in reversed(lines):
                if line.startswith("Result from applying"):
                    hypePars = line.strip()
                if line.startswith("Running the code took"):
                    time = re.search(r"Running the code took\s+(\d+\.\d+)", line).group(1)
                if line.startswith("ROC AUC:"):
                    rocAUC = re.search(r"ROC AUC:\s+(\d+\.\d+)", line).group(1)
                if line.startswith("F1 score:"):
                    fOne = re.search(r"F1 score:\s+(\d+\.\d+)", line).group(1)
                if line.startswith("Accuracy:"):
                    accuracy = re.search(r"Accuracy:\s+(\d+\.\d+)", line).group(1)
        else:
            # Do nothing if the last line does not start with "Accuracy"
            print("THE FOLLOWING FILE IS NOT COMPLETE" + fileName)
    
    hypePars_splitted = hypePars.split("-")
    if 'Class' not in hypePars_splitted[0]:
        print("The word 'Class' is not in the string.")
    else:
        const_C = hypePars_splitted[1].replace("C", "")
        if "p" in const_C:
            const_C = const_C.replace("p", ".")
        
        gamma = hypePars_splitted[2].replace("gamma", "")
        if "p" in gamma:
            gamma = gamma.replace("p", ".")
        
        weight = hypePars_splitted[3].replace("weight", "")
        
        trainSize = hypePars_splitted[4].replace("trainSize", "")
        
        testSize = hypePars_splitted[5].replace("testSize", "")
        
        foldIdx = hypePars_splitted[6].replace("foldIdx", "").replace(".pkl", "")
        
        print('c', const_C)
        print('gamma', gamma)
        print('weight', weight)
        print('trainsize', trainSize)
        print('testsize', testSize)
        print('foldidx', foldIdx)
        print('time', time)
        print('roc', rocAUC)
        print('f1', fOne)
        print('accuracy', accuracy)

        if (trainSize=='20000' and testSize=='5000'):
            indexOfnewData = ('RBF',const_C,gamma,weight)#Index in this order: kernel, C-Class, gamma, weight
            newData = pd.DataFrame({
                "TimeToRun-fold"+foldIdx: [time],
                "rocAUC-fold"+foldIdx: [rocAUC],
                "F1score-fold"+foldIdx: [fOne],
                "Accuracy-fold"+foldIdx: [accuracy]
                }, index=[indexOfnewData])
            print('newData=', newData)
            
            if indexOfnewData in df.index:
                df.update(newData)
                #print('UPDATED DF=', df)
            else:
                df = pd.concat([df, newData], ignore_index=False)
                #print('ADDED NEW ROW. RESULT IS:', df)
        else:
            print('TRAIN AND TEST SIZE NOT 20K AND 5K, RESPECTIVELY.')
    print('=========================================')
