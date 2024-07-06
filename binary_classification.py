import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import randint, uniform


def bin_clas_analysis(X, y, cv1=100, cv2=100):
    pipe = Pipeline([
        ('std', StandardScaler()),
        ('clf', XGBClassifier())
        ])
    regr_params = {
        "clf__n_estimators": randint(10, 200),
        "clf__gamma": uniform(0.0, 10.0),
        "clf__objective": ['binary:logistic'],
        "clf__booster": ['gbtree', 'dart'],
        }
    skfolds = StratifiedKFold(
        n_splits=cv1,
        random_state=0
        )
    rscv = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=regr_params,
        cv=StratifiedKFold(n_splits=cv2, random_state=0),
        scoring='accuracy',
        verbose=0,
        n_jobs=-1,
        n_iter=10000,
        refit=True,
        random_state=0,
        )
    tot_res_dic = dict()
    y_val_preds = list()

    iter_n = 1
    for tr_idx, val_idx in skfolds.split(X, y):
        X_tr = X.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]
        rscv.fit(X_tr, y_tr)
        y_val_pred = rscv.predict_proba(X_val)[:, 1].flatten()
        y_val_preds.append(pd.Series(y_val_pred))
        iter_n += 1
    tot_res_dic['preds_probas'] = pd.concat(y_val_preds)
    return tot_res_dic







