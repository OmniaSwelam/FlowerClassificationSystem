import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np
import features as ft
import svm, knn, qda, lda
from sklearn.metrics import classification_report
import RandomForest as rf
#%%Exploring the data
flowers = pd.read_csv('flower_images/flower_labels.csv')
flower_files = flowers['file']
flower_targets = flowers['label'].values

#%%
features=np.array([])
for i in range(flower_files.shape[0]):
    image= plt.imread("flower_images/"+flower_files[i])
    image= image[:,:,0:3]
    f= ft.calc_feat(image)
    features= np.vstack((features,f))  if features.size else f

x_train, x_test, y_train, y_test = train_test_split(features, flower_targets, 
                                                    test_size = 0.25, random_state = 1)

#%%classify:
#SVM    
accuracy, y_pred= svm.svm_classifier(x_train, x_test,y_train, y_test) 
print("Accuracy of SVM: ",accuracy)  
target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5','class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test, y_pred, target_names=target_names))  

#KNN  
accuracy1, y1_pred= knn.knn_classifier(x_train, x_test,y_train, y_test) 
print("Accuracy of KNN: ",accuracy1)  
target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5','class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test, y1_pred, target_names=target_names))    

#Random Forest
accuracy2, y2_pred= rf.RandomForest(x_train, x_test,y_train, y_test) 
print("Accuracy of Random Forest: ",accuracy2)  
target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5','class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test, y2_pred, target_names=target_names))    

#QDA
'''accuracy3, y3_pred= qda.qda_classifier(x_train, x_test,y_train, y_test) 
print("Accuracy of QDA: ",accuracy3)  
target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5','class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test, y3_pred, target_names=target_names)) '''

#LDA
accuracy4, y4_pred= lda.lda_classifier(x_train, x_test,y_train, y_test) 
print("Accuracy of LDA: ",accuracy4)  
target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5','class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test, y4_pred, target_names=target_names)) 