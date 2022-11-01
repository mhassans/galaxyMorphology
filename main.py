import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataframe():
    GZ1 = pd.read_csv('data/GalaxyZoo1/GalaxyZoo1_DR_table2.csv') #Galaxy zoo data from https://data.galaxyzoo.org/
    features = pd.read_csv('data/features/Barchi19_Morph-catalog_670k-galaxies.csv')
                    #from https://www.sciencedirect.com/science/article/pii/S2213133719300757?via%3Dihub#ec-research-data
    features = features.rename(columns={'dr7objid':'OBJID'})
    features = features.drop_duplicates()
    df = pd.merge(features, GZ1, on='OBJID', how='inner')
    df = df[df.UNCERTAIN==0].drop(columns=['UNCERTAIN']) #remove Uncertain category, i.e. not elliptical nor spiral
    return df

def main():
    df = prepare_dataframe()

if __name__=="__main__":
    main()
