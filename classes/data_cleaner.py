import numpy as np
import pandas as pd


def cleanFaces(df):
    """
    Cleans non faces from dataset
    if sum of attribues = -5 then image is not a face
    """
    ex = []

    for i in range(1,1+len(df['file_name'])):
        c =int(df['hair_color'][i])
        c = c + int(df['eyeglasses'][i])
        c = c + int(df['smiling'][i])
        c = c + int(df['young'][i])
        c = c + int(df['human'][i])
        if c == -5:
            ex.append(i-1)

    # update
    df = df.drop(df.index[ex])
    return df

def replaceNeg(df):
    """
    replaces -1 to 0 for the labels
    """
    mask = df[df.columns[1]] == -1
    column_name = df.columns[1]
    df.loc[mask, column_name] = 0
    return df

def removeNeg(df):
    """
    removes attributes with attribute -1 from dataset
    """
    ex = []
    for i in range(len(df['file_name'])):
        if df[df.columns[1]].iloc[i]==-1:
            ex.append(i)
    df = df.drop(df.index[ex])
    return df
