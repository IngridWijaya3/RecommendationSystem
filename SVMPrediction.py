import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import scipy
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pickle
import os.path
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


#user_item_combine.csv
cwd = os.getcwd()
filepath =cwd+"/user_item_combine.csv"
df = pd.read_csv(filepath)
df=shuffle(df)
train, test = train_test_split(df, test_size = 0.2)

X_train=np.array(train[['item_id_int' , 'user_also_view_int',  'brand_int','price']])
Y_train=np.array(train['user_buy_after_viewing_bool'])
X_test=np.array(test[['item_id_int' , 'user_also_view_int',  'brand_int','price']])
Y_test=np.array(test['user_buy_after_viewing_bool'])

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X_train)

X_train_imp = imp.transform(X_train)
X_train_scaled = preprocessing.scale(X_train_imp)
X_test_impt=imp.transform(X_test)
X_test_scaled = preprocessing.scale(X_test_impt)
#print(X_train_imp)
#print(X_train_scaled)

'''
kf = KFold(n_splits=10) # Define the split - into 2 folds

for train_index, test_index in kf.split(X_train_scaled):
     print("TRAIN:", train_index, "TEST:", test_index)
     xtrain, xtest = X_train_scaled[train_index], X_train_scaled[test_index]
     yrain, ytest = Y_train[train_index], Y_train[test_index]
'''
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

pipelinefilepath =cwd+"/gridsearchsvm.pkl"
my_file = Path(pipelinefilepath)

if my_file.is_file():
    
    clf = joblib.load(pipelinefilepath)
else:
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    clf.fit(X_train_scaled, Y_train)
    joblib.dump(clf, pipelinefilepath)

svmmodelfilepath =cwd+"/svmmodel.pkl"
svm_file = Path(svmmodelfilepath)
if svm_file.is_file():
    bestmodel = pickle.load(open(svmmodelfilepath, 'rb') , encoding='latin1')

else:
    svc=svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma)
    bestmodel = svc.fit(X_train_scaled, Y_train)
    pickle.dump(bestmodel, open(svmmodelfilepath, 'wb'))

'''
best_parameters=clf.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
        print ('\t%s: %r' % (param_name, best_parameters[param_name]))
        '''
print('Best score for data1:', clf.best_score_)
print('Best C:',clf.best_estimator_.C)
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
print('Prediction Score ',clf.score(X_test_scaled, Y_test))

y_score=bestmodel.score(X_test_scaled, Y_test)
print('Prediction with best model ', y_score)
Y_pred =bestmodel.predict(X_test_scaled)
y_score = bestmodel.decision_function(X_test_scaled)
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(Y_test, Y_pred)
df2 = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
'''
best_parameters=clf.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
        print ('\t%s: %r' % (param_name, best_parameters[param_name]))
print('Best score for data1:', clf.best_score_)
print('Best C:',clf.best_estimator_.C)
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)


#for i in range(2):
#    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
#   roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
''' 
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
'''
precision, recall, thresholds = precision_recall_curve(Y_test, y_score)
precision = dict()
recall = dict()
average_precision = dict()
for i in range(2):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly

precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
'''
conf_mat = confusion_matrix(Y_test, Y_pred)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=df.user_buy_after_viewing_bool.values, yticklabels=df.user_buy_after_viewing_bool.values)

tn, fp, fn, tp =confusion_matrix(Y_test, Y_pred).ravel()
print(tn)
print(fp)
print(fn)
print(tp)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')


fig2, ax4 = plt.subplots(figsize=(10, 10))
'''
ax4.step(recall, precision, color='b', alpha=0.2,
         where='post')
ax4.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_ylim([0.0, 1.05])
ax4.set_xlim([0.0, 1.0])
ax4.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
 '''
fig3, ax3 = plt.subplots(figsize=(10, 10))
lw = 2
aucvalue=auc(fpr,tpr)
ax3.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % aucvalue)
ax3.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Receiver operating characteristic example')
ax3.legend(loc="lower right")


plt.interactive(False)
plt.show()