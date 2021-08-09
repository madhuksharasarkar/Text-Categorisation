import pandas as pd
import string
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# Function to remove punctuations from text
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


#########################################################################################################
################################### Loading and Cleaning Data ###########################################
#########################################################################################################


# Loading training data
train_data = load_files('training', load_content=True, encoding = 'utf-8', shuffle = False)

train_df = pd.DataFrame(train_data.data)
train_df.columns = ['Documents']
train_df['Class'] = train_data.target

# Cleaning training data
train_df["Documents"] = train_df['Documents'].apply(remove_punctuations)
train_df["Documents"] = train_df["Documents"].str.lower()
train_df["Documents"] = train_df["Documents"].str.replace('\d+', '')
train_df["Documents"] = train_df["Documents"].str.split()
train_df["Documents"] = train_df["Documents"].apply(lambda x: [item for item in x if item not in stop_words])
train_df["Documents"] = train_df["Documents"].apply(lambda x: [ps.stem(y) for y in x])
train_df["Documents"] = train_df["Documents"].str.join(" ")


# Loading test data
test_data = load_files('test', load_content=True, encoding = 'utf-8', shuffle = False)

test_df = pd.DataFrame(test_data.data)
test_df.columns = ['Documents']
test_df['Class'] = test_data.target

# Cleaning test data
test_df["Documents"] = test_df['Documents'].apply(remove_punctuations)
test_df["Documents"] = test_df["Documents"].str.lower()
test_df["Documents"] = test_df["Documents"].str.replace('\d+', '')
test_df["Documents"] = test_df["Documents"].str.split()
test_df["Documents"] = test_df["Documents"].apply(lambda x: [item for item in x if item not in stop_words])
test_df["Documents"] = test_df["Documents"].apply(lambda x: [ps.stem(y) for y in x])
test_df["Documents"] = test_df["Documents"].str.join(" ")


#########################################################################################################
##################################### Data Splitting ####################################################
#########################################################################################################

trn_data = train_df["Documents"]
class_label_trn = train_df["Class"]

tst_data = test_df["Documents"]
class_label_tst = test_df["Class"]


#########################################################################################################
##################################### tf-idf Matrix #####################################################
#########################################################################################################

vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df = 3)
matrix = vectorizer.fit_transform(trn_data)


## Now we shall use different classifiers for training and testing the data, 
## and we shall print the confusion matrix, precision, recall and F-measure 
## for each classifier.

#########################################################################################################
################################### Multinomial Naive Bayes #############################################
#########################################################################################################

param_grid_mnb = [{'alpha':[0.001, 0.01, 0.1, 1]}]
mnb = MultinomialNB()

grid = grid_search.GridSearchCV(mnb,param_grid_mnb,cv=10,scoring='accuracy')
grid.fit(matrix,class_label_trn)
mnb = grid.best_estimator_
print ('\n The Multinomial Naive Bayes Classifier is as follows: \n')
print (grid.best_estimator_)

predicted_class_label_mnb = mnb.predict(vectorizer.transform(tst_data))
predicted_class_label_mnb = list(predicted_class_label_mnb)

print ('\n Confusion Matrix \n')
print (confusion_matrix(class_label_tst, predicted_class_label_mnb))
pr1=precision_score(class_label_tst, predicted_class_label_mnb, average='macro')
print ('\n Precision:'+str(pr1))
re1=recall_score(class_label_tst, predicted_class_label_mnb, average='macro')
print ('\n Recall:'+str(re1))
fm1=f1_score(class_label_tst, predicted_class_label_mnb, average='macro') 
print ('\n F-measure:'+str(fm1))


#########################################################################################################
######################################### Bernoulli Naive Bayes #########################################
#########################################################################################################

param_grid_bnb = [{'alpha':[0.001, 0.01, 0.1, 1.0]}]
bnb = BernoulliNB()

grid = grid_search.GridSearchCV(bnb,param_grid_bnb,cv=10,scoring='accuracy')
grid.fit(matrix,class_label_trn)
bnb = grid.best_estimator_
print ('\n The Bernoulli Naive Bayes Classifier is as follows: \n')
print (grid.best_estimator_)

predicted_class_label_bnb = bnb.predict(vectorizer.transform(tst_data))
predicted_class_label_bnb = list(predicted_class_label_bnb)

print ('\n Confusion Matrix \n')
print (confusion_matrix(class_label_tst, predicted_class_label_bnb))
pr2=precision_score(class_label_tst, predicted_class_label_bnb, average='macro')
print ('\n Precision:'+str(pr2))
re2=recall_score(class_label_tst, predicted_class_label_bnb, average='macro')
print ('\n Recall:'+str(re2))
fm2=f1_score(class_label_tst, predicted_class_label_bnb, average='macro') 
print ('\n F-measure:'+str(fm2))


#########################################################################################################
######################################## k-Nearest Neighbour ############################################
#########################################################################################################

param_grid_knn =[{'n_neighbors':list(range(1,22)),'metric':['euclidean','minkowski','cosine','manhattan']}]
knn = KNeighborsClassifier()

grid = grid_search.GridSearchCV(knn,param_grid_knn,cv=10,scoring='accuracy')
grid.fit(matrix,class_label_trn)
knn = grid.best_estimator_
print ('\n The KNN Classifier is as follows: \n')
print (grid.best_estimator_)

predicted_class_label_knn = knn.predict(vectorizer.transform(tst_data))
predicted_class_label_knn = list(predicted_class_label_knn)

print ('\n Confusion Matrix \n')
print (confusion_matrix(class_label_tst, predicted_class_label_knn))
pr3=precision_score(class_label_tst, predicted_class_label_knn, average='macro')
print ('\n Precision:'+str(pr3))
re3=recall_score(class_label_tst, predicted_class_label_knn, average='macro')
print ('\n Recall:'+str(re3))
fm3=f1_score(class_label_tst, predicted_class_label_knn, average='macro') 
print ('\n F-measure:'+str(fm3))


#########################################################################################################
######################################### Logistic Regression ###########################################
#########################################################################################################

param_grid_lgm = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
lgm = LogisticRegression()

grid = grid_search.GridSearchCV(lgm,param_grid_lgm,cv=10,scoring='accuracy')
grid.fit(matrix,class_label_trn)
lgm = grid.best_estimator_
print ('\n The Logistic Regression Classifier is as follows: \n')
print (grid.best_estimator_)

predicted_class_label_lgm = lgm.predict(vectorizer.transform(tst_data))
predicted_class_label_lgm = list(predicted_class_label_lgm)

print ('\n Confusion Matrix \n')
print (confusion_matrix(class_label_tst, predicted_class_label_lgm))
pr4=precision_score(class_label_tst, predicted_class_label_lgm, average='macro')
print ('\n Precision:'+str(pr4))
re4=recall_score(class_label_tst, predicted_class_label_lgm, average='macro')
print ('\n Recall:'+str(re4))
fm4=f1_score(class_label_tst, predicted_class_label_lgm, average='macro') 
print ('\n F-measure:'+str(fm4))


#########################################################################################################
######################################### Support Vector Machine ########################################
#########################################################################################################

param_grid_svm = [{'C': [0.001, 0.01, 0.1, 10, 100, 1000]}]
svm = SVC(kernel = 'linear')

grid = grid_search.GridSearchCV(svm,param_grid_svm,cv=10,scoring='accuracy')
grid.fit(matrix,class_label_trn)
svm = grid.best_estimator_
print ('\n The Support Vector Machine Classifier is as follows: \n')
print (grid.best_estimator_)

predicted_class_label_svm = svm.predict(vectorizer.transform(tst_data))
predicted_class_label_svm = list(predicted_class_label_svm)

print ('\n Confusion Matrix \n')
print (confusion_matrix(class_label_tst, predicted_class_label_svm))
pr5=precision_score(class_label_tst, predicted_class_label_svm, average='macro')
print ('\n Precision:'+str(pr5))
re5=recall_score(class_label_tst, predicted_class_label_svm, average='macro')
print ('\n Recall:'+str(re5))
fm5=f1_score(class_label_tst, predicted_class_label_svm, average='macro') 
print ('\n F-measure:'+str(fm5))
