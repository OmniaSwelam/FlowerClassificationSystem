from sklearn.metrics  import confusion_matrix
from sklearn.svm import SVC

def svm_classifier(train,test,train_label,test_label):
     #training a svm classifier

    svm_model_linear = SVC(kernel = 'linear', C = 1000).fit(train, train_label) #?
    svm_predictions = svm_model_linear.predict(test)
 
    # model accuracy for X_test  
    accuracySVM = svm_model_linear.score(test, test_label)
 
    # creating a confusion matrix
    cmSVM = confusion_matrix(test_label, svm_predictions)
    
    return accuracySVM , svm_predictions
