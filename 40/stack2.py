# coding:utf-8
# 另一个Stacking的例子
# https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking


if __name__ == "__main__":
    link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    names = ['Class', 'Alcohol', 'Malic acid', 'Ash',
         'Alcalinity of ash' ,'Magnesium', 'Total phenols',
         'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',     'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
         'Proline']
    df = pd.read_csv(link, header=None, names=names)
    print(df.sample(5))
    
    y = df[["Class"]]
    X = df.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    models = [
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=100, max_depth=3),
        XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3)]
    S_train, S_test = stacking(models, X_train, y_train, X_test, regression=False, mode='oof_pred_bag', needs_proba=False, save_dir=None, metric=accuracy_score, n_folds=4, stratified=True, shuffle=True, random_state=0, verbose=2)
    
    model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3)
    model = model.fit(S_train, y_train)
    y_pred = model.predict(S_test)
    print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
