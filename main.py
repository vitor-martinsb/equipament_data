import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
import graphviz
from sklearn import tree
import seaborn as sns

class equipament_analyse:
    def __init__(self, fold='./data/', file='O_G_Equipment_Data.xlsx'):
        self.data = pd.read_excel(fold+file, decimal=',')
        self.data = self.data[['Cycle', 'Preset_1', 'Preset_2', 'Temperature',
                               'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency', 'Fail']]

    def RandomForest_analyse(self, test_size=0.2):
        X = self.data[['Preset_1', 'Preset_2', 'Temperature', 'Pressure',
                       'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']].to_numpy()
        y = self.data['Fail'].to_numpy(dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)
        smote = SMOTE(sampling_strategy={0:  np.array(np.unique(
            y_train, return_counts=True)).T[0, 1], 1:  np.array(np.unique(y_train, return_counts=True)).T[0, 1]})
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier()
        rf.fit(X_train_res, y_train_res)
        y_pred = rf.predict(X_test)

        print(classification_report(y_test, y_pred))

    def RandomForest_analyse(self, test_size=0.2):
        X = self.data[['Temperature', 'Pressure', 'VibrationX',
                       'VibrationY', 'VibrationZ', 'Frequency']].to_numpy()
        y = self.data['Fail'].to_numpy(dtype=int)

        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=test_size)

        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier()
        rf.fit(X_train_res, y_train_res)
        y_pred = rf.predict(X_test)

        print(classification_report(y_test, y_pred))
        print("Acurácia: ", accuracy_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = 100 * (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_normalized, annot=True, fmt= ".2f" , cmap="plasma", cbar=False, annot_kws={"fontsize": 12,"fontweight":'bold'})
        plt.xlabel('Predicted',fontsize='14',fontweight='bold')
        plt.ylabel('Actual',fontsize='14',fontweight='bold')
        plt.xticks([0.5,1.5],['Operating','Fail'],fontsize=12)
        plt.yticks([0.5,1.5],['Operating','Fail'],fontsize=12)
        plt.title('Confusion Matrix using Random Forest (%)',fontsize=14,fontweight='bold')
        plt.show()


    def analyse_category_preset(self):

        data_aux = self.data

        data_aux['Fail'] = data_aux['Fail'].to_numpy(dtype=int)
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(data_aux['Cycle'].to_numpy(dtype=int),data_aux['Fail'].to_numpy(dtype=int),color='darkblue')
        ax.fill_between(data_aux['Cycle'].to_numpy(dtype=int),data_aux['Fail'].to_numpy(dtype=int), color='blue',alpha=0.5)
        plt.ylabel('Fails Detected',fontsize=14,fontweight='bold')
        plt.xlabel('Cycles',fontsize=14,fontweight='bold')
        plt.title('Operating Cycle',fontsize=14,fontweight='bold')
        ax.set_yticks([0,1], labels=['Operating','Fail'],fontsize=12)
        plt.xticks(fontsize=12)
        plt.show()

        data_aux['Preset_1'] = data_aux['Preset_1'].to_numpy(dtype=str)
        data_aux['Preset_2'] = data_aux['Preset_2'].to_numpy(dtype=str)
        data_aux['Preset Category'] ='PR1-'+data_aux['Preset_1'] + '| PR2-' + data_aux['Preset_2']
        data_aux[data_aux['Fail']==1]['Preset Category'].value_counts().plot(kind='bar')
        plt.ylabel('Amount',fontsize=14,fontweight='bold')
        plt.xlabel('Preset 1 and Preset 2',fontsize=14,fontweight='bold')
        plt.xticks(rotation=45)
        plt.title('Preset configurations combinations of Fails',fontsize=14,fontweight='bold')

        plt.show()

        for var in ['Temperature','Pressure','VibrationX','VibrationY','VibrationZ','Frequency']:
            fig, ax = plt.subplots()
            ax.plot(data_aux['Cycle'].to_numpy(dtype=int),500*data_aux['Fail'].to_numpy(dtype=int),color='darkred',alpha=0.5)
            ax.fill_between(data_aux['Cycle'].to_numpy(dtype=int),500*data_aux['Fail'].to_numpy(dtype=int),color='red',alpha=0.3)
            ax.plot(data_aux['Cycle'].to_numpy(dtype=int),data_aux[var].to_numpy(dtype=float),color='darkblue')
            plt.ylabel(var,fontsize=14,fontweight='bold')
            plt.xlabel('Cycles',fontsize=14,fontweight='bold')
            plt.xticks(rotation=45)
            plt.title(var + ' variation',fontsize=14,fontweight='bold')
            plt.ylim([0.5,max(data_aux[var].to_numpy(dtype=int))])
            plt.show()

    def tree_decision(self):

        data_aux = self.data
        X = data_aux[['Temperature', 'Pressure', 'VibrationX','VibrationY', 'VibrationZ', 'Frequency']]
        data_aux['Fail'] = data_aux['Fail'].to_numpy(dtype=int)
        y = data_aux['Fail']
        

        tree_class = DecisionTreeClassifier()
        tree_class.fit(X, y)

        importance = tree_class.feature_importances_
        df_importance = pd.DataFrame({'Variable': X.columns, 'Importance': importance})
        df_importance = df_importance.sort_values('Importance', ascending=True)
        df_importance['Importance'] = 100*df_importance['Importance'].to_numpy()
        df_importance.plot(x='Variable',y='Importance',kind='barh',color='#bc5090')
        plt.ylabel('Variable',fontsize=14,fontweight='bold')
        plt.xlabel('Importance (%)',fontsize=14,fontweight='bold')
        plt.title('Variable importance based on the decision tree',fontsize=14,fontweight='bold')
        plt.xticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.show()
        print(df_importance)

        dot_data = tree.export_graphviz(tree_class, out_file=None, 
                                        feature_names=['Temperature', 'Pressure', 'VibrationX','VibrationY', 'VibrationZ', 'Frequency'],  
                                        class_names='Fail',  
                                        filled=True, rounded=True,  
                                        special_characters=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_tree(tree_class, feature_names=['Temperature', 'Pressure', 'VibrationX','VibrationY', 'VibrationZ', 'Frequency'],ax=ax, filled=False, node_ids=True)
        plt.show()

        graph = graphviz.Source(dot_data)

        graph.view()

if __name__ == '__main__':
    ea = equipament_analyse()
    #ea.tree_decision()
    #ea.analyse_category_preset()
    ea.RandomForest_analyse()
    
    print('Finish')
