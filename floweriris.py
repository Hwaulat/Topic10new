import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True) 
    df = pd.read_csv(r'C:/TubesPF/IRIS.csv')

    df.shape
    df.head(20)
    df.describe()
    df['species'].value_counts()
    df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    df.hist()
    plt.show()
    sns.pairplot(df)
    plt.show()

    array = df.values
    X = array[:, 0:4]
    y = array[:, 4]
    xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=7)

    models = []
    # models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # Evaluate each model in turn
    results = []
    names = [] 
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, xtrain, ytrain, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    knn = KNeighborsClassifier()
    knn.fit(xtrain, ytrain)
    predictions = knn.predict(xval)
    print(accuracy_score(yval, predictions))
    print(classification_report(yval, predictions))
    plot_confusion_matrix(knn, xval, yval)