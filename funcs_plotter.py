import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def double_roc(df1, title1, df2, title2):
    fpr1, tpr1, _ = roc_curve(df1['trueLabels'], df1['scores'])
    roc_auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(df2['trueLabels'], df2['scores'])
    roc_auc2 = auc(fpr2, tpr2)    
    plt.figure()
    lw = 2
    plt.plot(fpr1,
             tpr1,
             color="darkgreen",
             lw=lw,
             label=title1+"ROC curve (area = %0.2f)" % roc_auc1,
            )    
    plt.plot(fpr2,
             tpr2,
             color="darkorange",
             lw=lw,
             label=title2+"ROC curve (area = %0.2f)" % roc_auc2,
            ) 
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('Classical vs Quantum')
    plt.legend(loc="lower right")
    plt.show()    
    
def single_roc(df, title):
    fpr, tpr, _ = roc_curve(df['trueLabels'], df['scores'])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr,
             tpr,
             color="darkorange",
             lw=lw,
             label="ROC curve (area = %0.2f)" % roc_auc,
            )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def renameOutputFile(folder):
    """
    Remove config# from file names. Not currently used. Used for renaming several files.
    """
    oldNames = []
    for file in os.listdir(folder):
        if fnmatch.fnmatch(file, '*config*'):
            oldNames.append(file)
    newNames = {}
    for oldName in oldNames:
        newName = oldName[:oldName.index('config')]+oldName[oldName.index('config')+8:]
        newNames[oldName] = newName
    for oldName, newName in newNames.items():
        os.rename(folder+oldName, folder+newName)
    
def getDataset(outputPath, classical, trainSize):
    """
    classical (bool): False for quantum-enhanced model, and True for classical.
    trainSize (int): train sample size
    """
    fileName = outputPath + '/' + 'result_'
    fileName = fileName + 'Class_' if classical else fileName + 'Quant_'
    fileName = fileName + 'trainSize' + str(trainSize) + '.pkl'
    return  pd.read_pickle(fileName)
