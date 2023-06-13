import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

class equipament_analyse:
    def __init__(self,fold='./data/',file = 'O_G_Equipment_Data.xlsx'):
        self.data = pd.read_excel(fold+file,decimal=',')
        self.data = self.data[['Cycle','Preset_1','Preset_2','Temperature','Pressure','VibrationX','VibrationY','VibrationZ','Frequency','Fail']]

    def RandomForest_analyse(self,test_size=0.2):
        X = self.data[['Preset_1','Preset_2','Temperature','Pressure','VibrationX','VibrationY','VibrationZ','Frequency']].to_numpy()
        y = self.data['Fail'].to_numpy(dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        smote = SMOTE(sampling_strategy={0:  np.array(np.unique(y_train, return_counts=True)).T[0,1], 1:  np.array(np.unique(y_train, return_counts=True)).T[0,1]})
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier()
        rf.fit(X_train_res, y_train_res)
        y_pred = rf.predict(X_test)

        print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    ea = equipament_analyse()
    ea.RandomForest_analyse()
    print('Finish')
