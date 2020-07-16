from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics  import confusion_matrix

def knn_classifier(train, test,train_label,test_label):

    knn = KNeighborsClassifier(n_neighbors=3).fit(train,train_label)
    
    #accuracy
    accuracyKNN =knn.score(test,test_label)
    
    ##creating confusion matrix
    knn_predictions =knn.predict(test)
    cmKNN=confusion_matrix(test_label,knn_predictions)

    
    return accuracyKNN,knn_predictions