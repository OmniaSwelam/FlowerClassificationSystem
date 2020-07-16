from sklearn.ensemble import RandomForestClassifier as randomForest

from sklearn.metrics  import confusion_matrix

def RandomForest(train, test,train_label,test_label):
    
    RF= randomForest(n_estimators=100, max_depth=2,
                              random_state=0).fit(train,train_label)
    #accuracy
    accuracyRF =RF.score(test,test_label)
    
    ##creating confusion matrix
    RF_predictions =RF.predict(test)
    cmKNN=confusion_matrix(test_label,RF_predictions)

    return accuracyRF,RF_predictions
