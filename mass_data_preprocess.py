import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# AME2003
masses03 = pd.read_fwf('mass.mas03', usecols=(2,3,4,11),
              names=('N', 'Z', 'A', 'avEbind'),
              widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
              header=39,
              index_col=False)
masses03['avEbind'] = pd.to_numeric(masses03['avEbind'], errors='coerce')
masses03 = masses03.dropna()
masses03 = masses03.drop(0)
masses03['BE_per_A'] = masses03['avEbind'] / 1000
masses03['AME2003'] = masses03['BE_per_A'] * masses03['A']
masses03["S2n_03"] = np.nan
masses03 = masses03.drop(columns = ['avEbind', 'BE_per_A'])

for index, row in masses03.iterrows():
    if np.any((masses03["N"] == (row["N"] - 2)) & (masses03["Z"] == row["Z"])):
        masses03.loc[(masses03["N"] == (row["N"])) & (masses03["Z"] == row["Z"]), "S2n_03"] = masses03[(masses03["N"] == (row["N"])) & (masses03["Z"] == row["Z"])]["AME2003"].squeeze()  -  masses03[(masses03["N"] == (row["N"] - 2)) & (masses03["Z"] == row["Z"])]["AME2003"].squeeze()  
masses03 = masses03.dropna()
masses03 =  masses03.loc[(masses03["N"] % 2) == 0,]
masses03 = masses03.loc[masses03["Z"] > 1]

# AME2016
masses16 = pd.read_fwf('mass16.txt', usecols=(2,3,4,11),
              names=('N', 'Z', 'A', 'avEbind'),
              widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
              header=39,
              index_col=False)
masses16['avEbind'] = pd.to_numeric(masses16['avEbind'], errors='coerce')
masses16 = masses16.dropna()
masses16 = masses16.drop(0)
masses16['BE_per_A'] = masses16['avEbind'] / 1000
masses16['AME2016'] = masses16['BE_per_A'] * masses16['A']
masses16["S2n_16"] = np.nan
masses16 = masses16.drop(columns = ['avEbind', 'BE_per_A'])

for index, row in masses16.iterrows():
    if np.any((masses16["N"] == (row["N"] - 2)) & (masses16["Z"] == row["Z"])):
        masses16.loc[(masses16["N"] == (row["N"])) & (masses16["Z"] == row["Z"]), "S2n_16"] = masses16[(masses16["N"] == (row["N"])) & (masses16["Z"] == row["Z"])]["AME2016"].squeeze()  -  masses16[(masses16["N"] == (row["N"] - 2)) & (masses16["Z"] == row["Z"])]["AME2016"].squeeze()  
masses16 = masses16.dropna()
masses16 =  masses16.loc[(masses16["N"] % 2) == 0,]
masses16 = masses16.loc[masses16["Z"] > 1]

masses_merge = pd.merge(masses16, masses03, on = ["N", "Z", "A"], how = "left")
# print(masses_merge) 
# observed that the first row was missing


missing_row = {'N': 4, 'Z': 2, 'A': 6, 'AME2016': 29.271114, 'S2n_16': 0.975454, 'AME2003': 29.268102, 'S2n_03': 0.972442}
# Inserting the new row at the top
masses_merge.loc[-1] = missing_row
masses_merge.index = masses_merge.index + 1  # Shift the index to avoid having a row with index -1
masses_merge = masses_merge.sort_index()
# print(masses_merge)

# Adding model evaluations
exp = ['exp']
models = ['SKM*', 'SKP', 'SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1']
xls = pd.ExcelFile('separations_energies_NEW2018.xls')
#xls = pd.ExcelFile('separations_energies_AME2003.xls')
data_raw = pd.read_excel(xls, header=[0,3], sheet_name = 'S2n')

data_raw_sub = data_raw.T.loc[[('-', 'Z'), ('-', 'N'), ('exp', 'measure'), ('exp', 'sd'), (models[0], 'mean'), (models[1], 'mean'),
               (models[2], 'mean'),(models[3], 'mean'),(models[4], 'mean'),(models[5], 'mean')]].T
data_raw_sub.columns = ['_'.join(col) for col in data_raw_sub.columns.values]
data_raw_sub.columns = ['Z', 'N'] + ['exp'] + ['sd'] + models
is_na = data_raw_sub[['exp'] + models].notna()
is_na = is_na.all(axis = 1)
data_raw_sub = data_raw_sub[is_na]
data_raw_sub = data_raw_sub.rename(columns={'SKM*': 'SKM'})

# This is the final complete set of separation energies to be used for training and testing
S2n = pd.merge(data_raw_sub, masses_merge, on = ["N", "Z"], how = "left")
# print(S2n)

# Final training and testing set of S2n observations
S2n_test = S2n[S2n['S2n_03'].isna()]
S2n_train = S2n[np.invert(S2n['S2n_03'].isna())]
S2n_test = S2n_test[np.invert(S2n_test["S2n_16"].isna())]
# print(S2n_test)
S2n_train.to_csv('S2n_train.csv', index=False)
S2n_test.to_csv('S2n_test.csv', index=False)
