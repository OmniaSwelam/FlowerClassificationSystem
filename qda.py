from sklearn.metrics  import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def qda_classifier(train, test,train_label,test_label):

    qda = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False,
                              store_covariances=None, tol=0.0001)
    qda_model = qda.fit(train, train_label)
    qda_predictions=qda_model.predict(test)
    
     # model accuracy for X_test  
    accuracyQDA = qda_model.score(test, test_label)
 
    # creating a confusion matrix
    cmQDA = confusion_matrix(test_label, qda_predictions)
    
    return accuracyQDA, accuracyQDA