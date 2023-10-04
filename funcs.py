import yaml
import ast
import pandas as pd
from sklearn.model_selection import KFold
from functools import reduce
import numpy as np
from pathlib import Path

def fix_ttype_feature(df, Ndigit=5):
    """
    TType is a mix of float and string. It also has a few values with '-' between digits, e.g. '5.292892456050-05'.
    This example could mean 5.29^+5, which is far from the range of other values! So such values are discarded here;
    i.e. TTypes with '-' in the last Ndigit are discarded.
    """
    df = df.astype({'TType': 'str'})
    df = df[~(df['TType'].str.find('-',-1*Ndigit)!=-1)] # remove TTypes with '-' in the last Ndigit
    df = df.astype({'TType': 'float'})
    return df

def prepare_dataframe(trainPlusTestSize, minOfK, signalLabel, excludedFeatures=[], balancedSampling=False):
    random_state = 1
    GZ1 = pd.read_csv('data/GalaxyZoo1/GZ1.csv') #Galaxy zoo data from data.galaxyzoo.org (unwanted columns removed)
    features = pd.read_csv('data/features/features.csv') 
                    #from sciencedirect.com/science/article/pii/S2213133719300757 (unwanted columns removed)
    features = features.rename(columns={'dr7objid':'OBJID'})
    features = features.drop_duplicates()
    df = pd.merge(features, GZ1, on='OBJID', how='inner')
    df = df[df.UNCERTAIN==0].drop(columns=['UNCERTAIN']) #remove Uncertain category, i.e. not elliptical nor spiral
    df = df[df.Error==0] #Keep successful CyMorph processes only. See kaggle.com/datasets/saurabhshahane/galaxy-classification
    df = df[(df['G2']>-6000) & (df['S']>-6000) & (df['A']>-6000) & (df['C']>-6000)] #discard outliers
    df = df.drop(['Error'], axis=1) 
    df = df.drop(excludedFeatures, axis=1) 
    df = fix_ttype_feature(df)
    df = df[df['K']>minOfK]
    if balancedSampling:
        sample_signal = df[df[signalLabel]==1].sample(n=trainPlusTestSize//2, random_state=random_state)
        sample_bkg = df[df[signalLabel]==0].sample(n=trainPlusTestSize//2, random_state=random_state)
        df = pd.concat([sample_signal, sample_bkg])
    else:
        df = df.sample(n=trainPlusTestSize, random_state=random_state)
    df = df.reset_index(drop=True)
    return df

def get_train_test(df, signalLabel, usedFeatures, n_splits=5, fold_idx=0):
    if fold_idx>=n_splits:
        print("ERROR: Fold index must be between 0 and n_splits-1.")
        exit()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    indices = list(kf.split(df))
    train_id, test_id = indices[fold_idx]
    train = df.iloc[train_id]
    test = df.iloc[test_id]
    extraInfo = ['OBJID', 'TType','K']
    train_data = train[usedFeatures]
    train_labels = train[signalLabel]
    train_extraInfo = train[extraInfo]
    test_data = test[usedFeatures]
    test_labels = test[signalLabel]
    test_extraInfo = test[extraInfo]
    return train_data, train_labels, test_data, test_labels, train_extraInfo, test_extraInfo

def normalize_data(df):
    return ((df-df.min())/(df.max()-df.min()))

def makeOutputFileName(classical, trainSize):
    fileName = 'Class' if classical else 'Quant'
    fileName = fileName +'_trainSize' + str(int(trainSize))
    return fileName

def produceConfig(config, fileName):
    confPath = 'config/'+ config['subDir'] + '/'
    if not Path(confPath).exists():
        Path(confPath).mkdir(parents=True)
    filePath = confPath+'autoConfigs_'+fileName+'.yaml'
    with open(filePath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    print('The following config file was produced:')
    print(filePath)
    return filePath

def setConfigName(config):
    fileName = ''
    if config['classical']:
        fileName = 'Class'
        fileName += '-C' + str(config['C_class']).replace('.','p')
        fileName += '-gamma' + str(config['gamma']).replace('.','p')
    else:
        fileName = 'Quant'
        fileName += '-alpha' + str(config['alpha']).replace('.','p')
        fileName += '-alphaCorr' + str(config['alphaCorr']).replace('.','p')
        fileName += '-C' + str(config['C_quant']).replace('.','p')
        fileName += '-entangleType' + config['entangleType']
        fileName += '-balancedSampling' + str(config['balancedSampling'])
        fileName += '-IBMdevice' if config['RunOnIBMdevice'] else '-Simulation'
        fileName += '-dataMapFunc'
        if (config['data_map_func'] is None):
            fileName += 'None'
        else:
            fileName += config['data_map_func'].__name__
        fileName += '-nShots'
        if (config['nShots'] is None):
            fileName += 'None'
        else:
            fileName += str(config['nShots'])
        fileName += '-interaction' + 'and'.join(config['interaction'])
    
    fileName += '-weight'
    if (config['class_weight'] == None): # Here, 'class' has nothing to do with classical vs quantum.
        fileName += 'None'
    elif (config['class_weight'] == 'balanced'):
        fileName += 'Balanced'
    else:
        print('ERROR in funcs.py -> setConfigName function: You need to change this part of the code to set the file name when weights are other than None or balanced.')
        exit()
    testSetFraction = 1./config['n_splits']
    trainSetFraction = 1 - testSetFraction
    fileName += '-trainSize' + str(int(config['trainPlusTestSize']*trainSetFraction))
    fileName += '-testSize' + str(int(config['trainPlusTestSize']*testSetFraction))
    fileName += '-foldIdx' + str(config['fold_idx'])
    fileName += '-minOfK' + str(config['minOfK'])
    return fileName

def addMeanAndStdDev(df, classical):
    """
    Adds mean and std dev of metrics (roc, f1 score, and accuracy) and code run time as new df columns.
    Adds index information (weight, kernel, etc.) as new df columns.
    inputs:
        df: dataframe
        classical (boolean): whether it is for classical or quantum kernel
    """
    times = ["TimeToRun-fold{}".format(i) for i in range(5)]
    rocAUCs = ["rocAUC-fold{}".format(i) for i in range(5)]
    fOnes = ["F1score-fold{}".format(i) for i in range(5)]
    accuracies = ["Accuracy-fold{}".format(i) for i in range(5)]

    df["TimeToRun-mean"] = df[times].mean(axis=1)
    df["TimeToRun-std"] = df[times].std(axis=1)
    df["rocAUC-mean"] = df[rocAUCs].mean(axis=1)
    df["rocAUC-std"] = df[rocAUCs].std(axis=1)
    df["fOne-mean"] = df[fOnes].mean(axis=1)
    df["fOne-std"] = df[fOnes].std(axis=1)
    df["accuracy-mean"] = df[accuracies].mean(axis=1)
    df["accuracy-std"] = df[accuracies].std(axis=1)
    df['rocAUC-relError'] = df["rocAUC-std"]/df["rocAUC-mean"]
    df['fOne-relError'] = df["fOne-std"]/df["fOne-mean"]
    df['accuracy-relError'] = df["accuracy-std"]/df["accuracy-mean"]
    if classical:
        #Add index to column. Example of index: "('RBF', '1000000.0', 'scale', 'None')"
        df = df.assign(kernel = df.index.map(lambda x: str(ast.literal_eval(x)[0]))) #"ast" changes string back to tuple
        df = df.assign(C_class = df.index.map(lambda x: float(ast.literal_eval(x)[1])))
        df = df.assign(gamma = df.index.map(lambda x: str(ast.literal_eval(x)[2])))
        df = df.assign(weight = df.index.map(lambda x: str(ast.literal_eval(x)[3])))
    else:
        df = df.assign(alpha = df.index.map(lambda x: str(ast.literal_eval(x)[0]))) #"ast" changes string back to tuple
        df = df.assign(C_quant = df.index.map(lambda x: float(ast.literal_eval(x)[1])))
        df = df.assign(dataMapFunc = df.index.map(lambda x: str(ast.literal_eval(x)[2])))
        df = df.assign(interaction = df.index.map(lambda x: str(ast.literal_eval(x)[3])))
        df = df.assign(weight= df.index.map(lambda x: str(ast.literal_eval(x)[4])))
    return df

def dataMap_custom1(x):
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m*n, x)
    return coeff

def dataMap_custom2(x):
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m*n, 1 - x)
    return coeff

def dataMap_custom3(x):
    """
    x: array
    returns \prod_{i,j}(x[i]-x[j]) for all i & j in indices where i<j. If len(x) is 1 then returns x[0].
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, [(x[i] - x[j]) for i in range(len(x)) for j in range(i + 1, len(x))])
    return coeff

def dataMap_custom4(x):
    coeff = x[0] if len(x) == 1 else (1 - dataMap_custom3(x))
    return coeff

def get_corr(i, j, method='pearson'):
    """
    Correlation between features in galaxy data. Hard-coded now; should be cleaned later. 
    All 5 galaxy features must be present, sorted by ['C', 'H', 'G2', 'S', 'A'].
    corr is the correlation between features.
    """
    corr_pearson = np.array([
                             [ 1.        , -0.65134493, -0.58424253, -0.51686424,  0.23973728],
                             [-0.65134493,  1.        ,  0.9213353 ,  0.67520741, -0.20136038],
                             [-0.58424253,  0.9213353 ,  1.        ,  0.69492266, -0.15698898],
                             [-0.51686424,  0.67520741,  0.69492266,  1.        ,  0.05297785],
                             [ 0.23973728, -0.20136038, -0.15698898,  0.05297785,  1.        ]
                           ]) 
    corr_kendall = np.array([
                             [ 1.        , -0.48515647, -0.41004319, -0.33879072,  0.15136111],
                             [-0.48515647,  1.        ,  0.80436534,  0.35744024, -0.16003525],
                             [-0.41004319,  0.80436534,  1.        ,  0.30361616, -0.11764175],
                             [-0.33879072,  0.35744024,  0.30361616,  1.        ,  0.10885891],
                             [ 0.15136111, -0.16003525, -0.11764175,  0.10885891,  1.        ]
                           ])
    corr_spearman = np.array([
                              [ 1.        , -0.66689428, -0.58026856, -0.4882942 ,  0.23331018],
                              [-0.66689428,  1.        ,  0.94097503,  0.50798079, -0.24263426],
                              [-0.58026856,  0.94097503,  1.        ,  0.44270187, -0.17903032],
                              [-0.4882942 ,  0.50798079,  0.44270187,  1.        ,  0.16497113],
                              [ 0.23331018, -0.24263426, -0.17903032,  0.16497113,  1.        ]
                            ])
    if method == 'pearson':
        return corr_pearson[i,j]
    elif method == 'kendall':
        return corr_kendall[i,j]
    elif method == 'spearman':
        return corr_spearman[i,j]
    else:
        raise valueError("Method for calculating correlation not found")
