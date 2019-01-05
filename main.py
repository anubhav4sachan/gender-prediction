from data import getDataset, getLabels, getLangData
from tuning import svmTuner
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def getAccScore(truth, predicted):
    return accuracy_score(truth, predicted)

def data():
    labels = getLabels('train_answers.csv')
    [X_, y] = getDataset('train.csv', labels)
    X = X_[:, 4:]
    [Xtest_, ytest] = getDataset('test.csv', labels)
    Xtest = Xtest_[:, 4:]
    return [X, y, Xtest, ytest]

def langData():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    answers = pd.read_csv('train_answers.csv')
    features_labels = list(train.columns.values[5:]) 
    X = train[features_labels].values
    Xtest = test[features_labels].values
    y_ = answers['male'].values
    y = y_[0:200]
    ytest = y_[200:]
    y = np.repeat(y, 4, axis = 0)
    ytest = np.repeat(ytest, 4, axis = 0)

    [X_arabic, X_english] = getLangData(X)
    [y_arabic, y_english] = getLangData(y)
    [X_arabic_test, X_english_test] = getLangData(Xtest)
    [y_arabic_test, y_english_test] = getLangData(ytest)
    ytest = y_english_test
    
    return [X_arabic, y_arabic, X_arabic_test, X_english, y_english, X_english_test, ytest]
    
def classifierScore(X, y, Xtest, ytest):
    #LOGISTIC REGRESSION
    lg = LogisticRegression(C = 1e-10)
    lg.fit(X, y)
    y_predicted = lg.predict(Xtest)
    print("Logistic Regression\nAccuracy Score: ",  getAccScore(ytest, y_predicted))
    
    lg = LogisticRegression(C = 150)
    lg.fit(X, y)
    y_predicted = lg.predict(Xtest)
    print("Logistic Regression (with optimum Regularisation)")
    print("Accuracy Score: ",  getAccScore(ytest, y_predicted))
    
    #SUPPORT VECTOR MACHINES (SVC)
    svm = SVC(C = 5000, gamma = 0.005, probability = True)
    svm.fit(X, y)
    y_predicted = svm.predict(Xtest)
    print("Support Vector Machines (Classifier)\nAccuracy Score: ", getAccScore(ytest, y_predicted))
    
    #DECISION TREES CLASSIFIER
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    y_predicted = dt.predict(Xtest)
    print("Decision Tree Classifier\nAccuracy Score: ", getAccScore(ytest, y_predicted))
    
def driver():
    [X, y, Xtest, ytest] = data()
    [X_arabic, y_arabic, X_arabic_test, X_english, y_english, X_english_test, y_test] = langData()
    classifierScore(X, y, Xtest, ytest)
    print("\nArabic")
    classifierScore(X_arabic, y_arabic, X_arabic_test, y_test)
    print("\nEnglish")
    classifierScore(X_english, y_english, X_english_test, y_test)
    
if __name__ == "__main__":
    driver()