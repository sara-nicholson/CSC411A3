'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
    

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def KNN(train_data,test_data,n):
    from sklearn.neighbors import KNeighborsClassifier
    text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
                         ('clf',KNeighborsClassifier(n_neighbors=n))])
    text_clf.fit(train_data.data,train_data.target)
    predicted_test = text_clf.predict(test_data.data)
    predicted_train = text_clf.predict(train_data.data)
    print("KNN, n= {}, Train accuracy: {}".format(n,np.mean(predicted_train == train_data.target)))
    print("KNN, n={}, Test accuracy: {}".format(n,np.mean(predicted_test == test_data.target)))


def SVM(train_data,test_data):
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    params = {'clf__learning_rate':('optimal','invscaling'), 'clf__alpha':(0.01,0.02)}

    text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
                         ('clf',SGDClassifier(loss='hinge',penalty='l2', alpha = 0.01,
                                              random_state=42))])
#    gs_clf = GridSearchCV(text_clf,params,n_jobs=-1)
#    gs_clf.fit(train_data.data,train_data.target)
#    predicted_test = gs_clf.predict(test_data.data)
#    predicted_train = gs_clf.predict(train_data.data)
    text_clf.fit(train_data.data,train_data.target)
    predicted_test = text_clf.predict(test_data.data)
    predicted_train = text_clf.predict(train_data.data)
    print("SVM Train accuracy: {}".format(np.mean(predicted_train == train_data.target)))
    print("SVM Test accuracy: {}".format(np.mean(predicted_test == test_data.target)))
#    for param_name in sorted(params.keys()):
#        print("%s: %r" %(param_name,gs_clf.best_params_[param_name]))
def logistic(train_data,test_data):
    from sklearn.linear_model import LogisticRegression
    params = {'clf__solver':('sag','newton-cg','lbfgs')}
    text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
                         ('clf',LogisticRegression(penalty='l2',solver='sag',multi_class='multinomial'))])
    #gs_clf = GridSearchCV(text_clf,params,n_jobs=-1)
    #gs_clf.fit(train_data.data,train_data.target)
    #predicted_test = gs_clf.predict(test_data.data)
    #predicted_train = gs_clf.predict(train_data.data)
    text_clf.fit(train_data.data,train_data.target)
    predicted_test = text_clf.predict(test_data.data)
    predicted_train = text_clf.predict(train_data.data)
    #for param_name in sorted(params.keys()):
    #    print("%s: %r" %(param_name,gs_clf.best_params_[param_name]))
    print("Logistic Train accuracy: {}".format(np.mean(predicted_train == train_data.target)))
    print("Logistic Test accuracy: {}".format(np.mean(predicted_test == test_data.target)))
    
def multiNB(train_data,test_data):
    
    params = {'tfidf__use_idf':(True,False),'clf__alpha':(0.0125,0.015)}
    from sklearn.naive_bayes import MultinomialNB
    text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
                         ('clf',MultinomialNB(alpha=0.015))])
    
    #gs_clf = GridSearchCV(text_clf,params,n_jobs=-1)
    #gs_clf.fit(train_data.data,train_data.target)
    #predicted_test = gs_clf.predict(test_data.data)
    #predicted_train = gs_clf.predict(train_data.data)
    #for param_name in sorted(params.keys()):
    #    print("%s: %r" %(param_name,gs_clf.best_params_[param_name]))
    text_clf.fit(train_data.data,train_data.target)
    predicted_test = text_clf.predict(test_data.data)
    predicted_train = text_clf.predict(train_data.data)
    print("MultinomialNB Train accuracy: {}".format(np.mean(predicted_train == train_data.target)))
    print("MultinomialNB Test accuracy: {}".format(np.mean(predicted_test == test_data.target)))
    return predicted_test

def confMatrix(data,predict):
    
    trueLabels = data.target
    labelNames = data.target_names
    numTargets=  len(labelNames)
    confusled = np.zeros((numTargets,numTargets))
    for i in range(len(trueLabels)):
        confusled[predict[i]][trueLabels[i]] += 1
            
    return confusled

if __name__ == '__main__':
    train_data, test_data = load_data()
    #train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    #bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    
    #train_tfidf, test_tfidf, feature_names_tfidf = tf_idf_features(train_data, test_data)

    
#    
#    SVM(train_data,test_data)
#    logistic(train_data,test_data)
    test_predict = multiNB(train_data,test_data)
    confused = confMatrix(test_data,test_predict)
    
    import pandas as pd
    print(pd.DataFrame(confused, columns=[x for x in test_data.target_names]))
#    KNN(train_data,test_data,1)
#    KNN(train_data,test_data,5)
    #randomForest(train_data,test_data)