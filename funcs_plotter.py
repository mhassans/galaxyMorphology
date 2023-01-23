import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns

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
    plt.title('')
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

#def renameOutputFile(folder):
#    """
#    Remove config# from file names. Not currently used. Used for renaming several files.
#    """
#    oldNames = []
#    for file in os.listdir(folder):
#        if fnmatch.fnmatch(file, '*config*'):
#            oldNames.append(file)
#    newNames = {}
#    for oldName in oldNames:
#        newName = oldName[:oldName.index('config')]+oldName[oldName.index('config')+8:]
#        newNames[oldName] = newName
#    for oldName, newName in newNames.items():
#        os.rename(folder+oldName, folder+newName)
    
#def getDataset(outputPath, classical, trainSize):
#    """
#    classical (bool): False for quantum-enhanced model, and True for classical.
#    trainSize (int): train sample size
#    """
#    fileName = outputPath + '/' + 'result_'
#    fileName = fileName + 'Class_' if classical else fileName + 'Quant_'
#    fileName = fileName + 'trainSize' + str(trainSize) + '.pkl'
#    return  pd.read_pickle(fileName)

def getEfficiency(df):
    predictedAndTrueSize = len(df[(df['predictedLabels']==1)&(df['trueLabels']==1)])
    trueSize = len(df[df['trueLabels']==1])
    return round(predictedAndTrueSize/trueSize, 2)

def getPurity(df):
    predictedAndTrueSize = len(df[(df['predictedLabels']==1)&(df['trueLabels']==1)])
    predictedSize = len(df[df['predictedLabels']==1])
    return round(predictedAndTrueSize/predictedSize, 2)

def plotChart(dictClass, dictQuant, yLabel):
    
    labels = list(dictClass.keys())    
    
    SMALL_SIZE = 19
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 29
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title    
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(10,8))
    rects1 = ax.bar(x - width/2, list(dictClass.values()), width, label='Classical')
    rects2 = ax.bar(x + width/2, list(dictQuant.values()), width, label='Quantum-enhanced')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yLabel)
    ax.set_xlabel('training size')
    #ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower center')
    ax.set_ylim(0,1)
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, normalize_by_yTrue=False, normalize_by_yPredicted=False, title="", yTitle='true', xTitle='predicted'):
    cm = confusion_matrix(y_true, y_pred)
    if normalize_by_yTrue and normalize_by_yPredicted:
        print('ERROR: Cannot normalize by row and column at the same time!')
        sys.exit(1)
    if normalize_by_yTrue:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize_by_yPredicted:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    print (cm)
    
    #fontsize=20
    SMALL_SIZE = 19
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 29
    
    plt.figure(figsize=(12,10))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    #plt.rcParams.update({'font.size': 8})
    #plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(label = title)

    classes = ['elliptical', 'spiral']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='w' if cm[i, j] > thresh else 'k')

    plt.tight_layout(pad=1.4)
    #plt.rcParams.update({'font.size': 14})
    plt.ylabel(yTitle)
    plt.xlabel(xTitle)
    plt.clim(0,1)

def scatterplotWithErrors(x, y, x_err, y_err, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    plt.show()

def scatterplotColored(x, y, hue, data, palette, legendOutsideGraph=False, loglog=False):
    sns.scatterplot(x=x, y=y, hue=hue, data=data, palette=palette)
    if legendOutsideGraph:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
