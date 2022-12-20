# General Imports

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# Automatically Importing All Classifiers

estimators = all_estimators(type_filter='classifier')

all_class = []
all_class_names = []
for name, Classifiers in estimators:
    try:
        if name != 'GaussianProcessClassifier' and name != 'DummyClassifier':
            print('Appending', name)
            reg = Classifiers()
            all_class.append(reg)
            all_class_names.append(name)
    except Exception as e:
        print(e)

print(all_class)
print(all_class_names)

# Load and Describe Data

def load_pulsar_data():
    csv_path = os.path.abspath("HTRU_2.csv")
    return pd.read_csv(csv_path)

pulsar = load_pulsar_data()
print(pulsar.describe())

print(pulsar.corr())
print('*'*100)


# Train/Test Split and Preprocess Data
pulsar["EKIP_cat"] = pd.cut(pulsar["EKIP"],bins=[-2.0,0.027098,0.223240,0.473325,np.inf],labels=[1,2,3,4],right=True)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(pulsar,pulsar["EKIP_cat"]):
    train_set = pulsar.loc[train_index]
    test_set = pulsar.loc[test_index]

for set_ in(train_set,test_set):
    set_.drop("EKIP_cat",axis=1,inplace=True)

ptrain = train_set.copy()
ptest = test_set.copy()

ptrain_attrib = ptrain.drop("Class",axis=1)
ptrain_labels = ptrain["Class"].copy()
ptest_attrib = ptest.drop("Class",axis=1)
ptest_labels = ptest["Class"].copy()

scaler = StandardScaler()
ptrain_attrib = scaler.fit_transform(ptrain_attrib)
ptest_attrib = scaler.fit_transform(ptest_attrib)

pulsar.hist(bins=50, figsize=(15,11))
plt.savefig('variables.png')


# Simultaneous Run

def comparison(models,model_names):
    cv_data = []
    errors = []
    passed_models = []
    for i in range(len(models)):
        x = run(models[i])
        if type(x) == dict:
            cv_data += [x]
        else:
            errors += [models[i]]
    for j in range(len(models)):
        if models[j] not in errors:
            passed_models += [model_names[j]]
    figs = [test_best(cv_data, passed_models), box_acc(cv_data, passed_models), box_prec(cv_data, passed_models), box_rec(cv_data, passed_models), runtime(cv_data, passed_models)]
    for k in range(len(figs)):
        figs[k].savefig(f'fig_{k}.png',bbox_inches='tight')
    return test_best(cv_data, passed_models)

def run(model):
    print(f"checking {model}")
    try:
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=2)
        cv_output_dict = cross_validate(model, ptrain_attrib, ptrain_labels, scoring=["accuracy","precision","recall"], cv=cv_outer, return_estimator=True)
        return cv_output_dict
    except:
        pass


def runtime(cv_data, passed_models):
    timefig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(i[('fit_time')])
    sorted_index = df.median().sort_values().index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values(ascending=False).index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel('Run Time')
    plt.ylabel('Models')
    return timefig


def box_acc(cv_data, passed_models):
    accfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(i['test_accuracy'])
    sorted_index = df.median().sort_values(ascending=False).index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values().index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV Accuracy')
    return accfig


def box_prec(cv_data, passed_models):
    precfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(i['test_precision'])
    sorted_index = df.median().sort_values(ascending=False).index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values().index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV Precision')
    return precfig


def box_rec(cv_data, passed_models):
    recfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(i['test_recall'])
    sorted_index = df.median().sort_values(ascending=False).index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values().index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV Recall')
    return recfig


def test_best(cv_data, passed_models):
    acc = []
    prec = []
    rec = []
    print(cv_data)
    for i in cv_data:
        x = list(i['test_accuracy'])
        y = list(i['estimator'])
        for j in range(len(x)):
            if x[j] == max(x):
                best = y[j]
        predictions = best.predict(ptest_attrib)
        acc += [round(accuracy_score(ptest_labels,predictions),4)]
        prec += [round(precision_score(ptest_labels,predictions),4)]
        rec += [round(recall_score(ptest_labels,predictions),4)]
    print(acc)
    print(prec)
    print(rec)
    columnnames = ['Accuracy','Precision','Recall']
    df = pd.DataFrame(np.array([acc,prec,rec]).T,index=passed_models,columns=columnnames)
    sorted_df = df.sort_values(by='Accuracy',ascending=False)
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=sorted_df.values, rowLabels=sorted_df.index, colLabels=sorted_df.columns, loc='center')
    fig.tight_layout()
    return fig


y = all_class
y_names = all_class_names
x = all_class[0:5]
x_names = all_class_names[0:5]
comparison(y,y_names)


