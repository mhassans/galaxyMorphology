import glob
import pandas as pd
import re
import sys

#SET THE PATH OF THE FILES HERE
resultsFilesPath = './jobsOutput/withoutTtype/minOfK5/quantumKernel/Oct2023/pearson/'
#resultsFilesPath = './jobsOutput/withoutTtype/minOfK5/classicalKernel/RBF/'

df = pd.read_csv("hyperParOptResults.csv")#An empty df with column names

for fileName in glob.glob(resultsFilesPath + '*.o*'):
    with open(fileName, "r") as file:
        print(fileName)
        hypePars = ''
        time = ''
        rocAUC = ''
        fOne = ''
        accuracy = ''
        
        lines = file.readlines()
        if not lines:
            print("THE FOLLOWING FILE IS EMPTY" + fileName)
        # Check if the last line starts with "Accuracy"
        elif lines[-1].startswith("Accuracy"):
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
    
    if lines:
        hypePars_splitted = hypePars.split("-")
        if ('Quant' in hypePars_splitted[0]) and ('Class' not in hypePars_splitted[0]):
            alpha = hypePars_splitted[1].replace("alpha", "")
            if "p" in alpha:
                alpha = alpha.replace("p", ".")
            alphaCorr = hypePars_splitted[2].replace("alphaCorr", "")
            if "p" in alphaCorr:
                alphaCorr = alphaCorr.replace("p", ".")
            const_C = hypePars_splitted[3].replace("C", "")
            if "p" in const_C:
                const_C = const_C.replace("p", ".")
            entangleType = hypePars_splitted[4].replace("entangleType", "")
            balancedSampling = hypePars_splitted[5].replace("balancedSampling", "")
            simulation = hypePars_splitted[6].replace("Simulation", "")
            dataMapFunc = hypePars_splitted[7].replace("dataMapFunc", "")
            nShots = hypePars_splitted[8].replace("nShots", "")
            interaction = hypePars_splitted[9].replace("interaction", "")
            weight = hypePars_splitted[10].replace("weight", "")
            trainSize = hypePars_splitted[11].replace("trainSize", "")
            testSize = hypePars_splitted[12].replace("testSize", "")
            foldIdx = hypePars_splitted[13].replace("foldIdx", "")
            minOfK = hypePars_splitted[14].replace("minOfK", "").replace(".pkl", "")
            if (trainSize=='20000' and testSize=='5000'):
                #Index in this order: alpha, C-quant, dataMapFunc, interaction, weight
                indexOfnewData = (alpha,const_C,dataMapFunc,interaction,weight,alphaCorr)
                newData = pd.DataFrame({
                    "TimeToRun-fold"+foldIdx: [time],
                    "rocAUC-fold"+foldIdx: [rocAUC],
                    "F1score-fold"+foldIdx: [fOne],
                    "Accuracy-fold"+foldIdx: [accuracy]
                    }, index=[indexOfnewData])
                
                if indexOfnewData in df.index:
                    df.update(newData)
                else:
                    df = pd.concat([df, newData], ignore_index=False)
            else:
                print('TRAIN AND TEST SIZE NOT 20K AND 5K, RESPECTIVELY.')

        elif 'Class' in hypePars_splitted[0]:
            const_C = hypePars_splitted[1].replace("C", "")
            if "p" in const_C:
                const_C = const_C.replace("p", ".")
            gamma = hypePars_splitted[2].replace("gamma", "")
            if "p" in gamma:
                gamma = gamma.replace("p", ".")
            weight = hypePars_splitted[3].replace("weight", "")
            trainSize = hypePars_splitted[4].replace("trainSize", "")
            testSize = hypePars_splitted[5].replace("testSize", "")
            foldIdx = hypePars_splitted[6].replace("foldIdx", "")
            minOfK = hypePars_splitted[7].replace("minOfK", "").replace(".pkl", "")
            
            #print('c', const_C)
            #print('gamma', gamma)
            #print('weight', weight)
            #print('trainsize', trainSize)
            #print('testsize', testSize)
            #print('foldidx', foldIdx)
            #print('time', time)
            #print('roc', rocAUC)
            #print('f1', fOne)
            #print('accuracy', accuracy)

            if (trainSize=='20000' and testSize=='5000'):
                indexOfnewData = ('RBF',const_C,gamma,weight)#Index in this order: kernel, C-Class, gamma, weight
                newData = pd.DataFrame({
                    "TimeToRun-fold"+foldIdx: [time],
                    "rocAUC-fold"+foldIdx: [rocAUC],
                    "F1score-fold"+foldIdx: [fOne],
                    "Accuracy-fold"+foldIdx: [accuracy]
                    }, index=[indexOfnewData])
                
                if indexOfnewData in df.index:
                    df.update(newData)
                else:
                    df = pd.concat([df, newData], ignore_index=False)
            else:
                print('TRAIN AND TEST SIZE NOT 20K AND 5K, RESPECTIVELY.')
        else:
            print('CHECK THE FILE: IT IS NEITHER QUANTUM KERNEL NOR CLASSICAL.')
    else:
        print("EMPTY FILE")
    print('=========================================')

df.to_csv(resultsFilesPath + "hyperParOptResults2.csv", index=True)
