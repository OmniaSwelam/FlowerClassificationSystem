from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics  import confusion_matrix

def lda_classifier(train, test,train_label,test_label):

    lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001).fit(train,train_label)
    #accuracy
    accuracyLDA=lda.score(test, test_label)
    
    # creating a confusion matrix
    lda_predictions =lda.predict(test)
    cmLDA = confusion_matrix(test_label, lda_predictions)
    
    return accuracyLDA, lda_predictions

    

