from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn import svm
from sklearn.datasets import load_wine
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid


def plot_boundaries(ax, x_train, x_points, y_points, clf):
    x1 = x_train[:, 0]
    x2 = x_train[:, 1]
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, .02), np.arange(x2_min, x2_max, .02))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), alpha=0.9)
    ax.scatter(x_points[:, 0], x_points[:, 1], c=y_points, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), edgecolor='k', s=15)
    return ax

def shuffleSplitScale(dataset, i, j, seed):
    # READ AND SHUFFLE THE DATA
    x = dataset.data[:, (i, j)]
    y = dataset.target
    x, y = shuffle(x, y, random_state=seed)

    # SPLIT THE DATASET IN TRAIN AND TEST SETS
    v = int(len(list(y)) * 0.7)
    x_train_val = x[:v, :]
    x_test = x[v:, :]
    y_train_val = y[:v]
    y_test = y[v:]

    # SCALE ALL DATA USING ONLY TRAIN SET INFORMATION
    scaler = preprocessing.StandardScaler().fit(x_train_val)
    x_train_val = scaler.transform(x_train_val)
    x_test = scaler.transform(x_test)
    return x_train_val, x_test, y_train_val, y_test


# LOAD DATASET AND CREATE TRAIN/VALIDATION/TEST
wines = load_wine()
x_train_val, x_test, y_train_val, y_test = shuffleSplitScale(wines, 0, 1, 0)

t = int((len(x_train_val)+len(x_test))*0.5)
# SPLIT AGAIN TRAIN SET IN TRAIN AND VALIDATION
x_train = x_train_val[:t, :]
x_validation = x_train_val[t:, :]
y_train = y_train_val[:t]
y_validation = y_train_val[t:]

# PLOT ONLY THE ELEMENTS IN TRAIN SET TO SEE WHICH POINT WILL BE USED TO TRAIN THE MODEL AND CREATE THE BOUNDARIES
figure, ax = plt.subplots()
x1 = x_train[:, 0]
x2 = x_train[:, 1]
ax.scatter(x1, x2, c=y_train, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), edgecolor='k', s=15)
#plt.ylim(-1.9, 3.9)
#plt.xlim(-2.9, 3.5)
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.show()

# KNN
top = 0
ks = [1, 3, 5, 7]
weights = ['uniform', 'distance']
for w in weights:
    accuracies = []
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(11.5, 7.5)
    for i in range(len(ks)):
        clf = neighbors.KNeighborsClassifier(ks[i], weights=w)
        clf.fit(x_train, y_train)
        result = clf.predict(x_validation)
        accuracy = accuracy_score(y_validation, result)
        accuracies.append(accuracy*100)
        if top < accuracy:
            top = accuracy
            best_k = ks[i]
            best_weight = w
            best_clf = clf
        # PLOT DECISION BOUNDARIES FOR EACH TESTED VALUE OF K
        ax = axes[int(i/2)][i % 2]
        ax = plot_boundaries(ax, x_train, x_validation, y_validation, clf)
        ax.set_title("K = %i, weight = '%s'" % (ks[i], w))
    plt.show()
    print("Accuracies for " + w + " weight: " + str(accuracies))
    # PLOT A GRAPH THAT SHOW HOW THE ACCURACY ON THE VALIDATION SET VARIES WHEN CHANGING PARAMETER C
    plt.plot(ks, accuracies)
    plt.plot(ks, accuracies, 'bo')
    plt.xlabel('K value')
    plt.ylabel('% accuracy')
    plt.title('Accuracy with ' + w + ' weight depending on K value')
    plt.show()
print("Best accuracy with is " + str(top*100) + "%, calculated with " + best_weight + " weight and K = " + str(best_k))

# CALCULATE THE ACCURACY ON THE TEST SET
result = best_clf.predict(x_test)
accuracy = accuracy_score(y_test, result)
print("Accuracy calculated on test set "
      "with " + best_weight + " weight and K = " + str(best_k) + " is " + str(accuracy * 100) + "%.")
print()


# SVM
# TUNE C PARAMETER FOR LINEAR KERNEL AND RBF KERNEL
kernels = ['linear', 'rbf']
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for kernel in kernels:
    top = 0
    accuracies = []
    fig, axes = plt.subplots(4, 2)
    fig.set_size_inches(10, 14.5)
    for i in range(len(C_values)):
        c = C_values[i]
        clf = svm.SVC(C=c, kernel=kernel, gamma='auto')
        clf.fit(x_train, y_train)
        result = clf.predict(x_validation)
        accuracy = accuracy_score(y_validation, result)
        accuracies.append(accuracy*100)
        if top < accuracy:
            top = accuracy
            best_c = c
            best_clf = clf
        # PLOT DECISION BOUNDARIES FOR EACH TESTED VALUE OF PARAMETER C
        ax = axes[int(i/2)][i % 2]
        ax = plot_boundaries(ax, x_train, x_validation, y_validation, clf)
        ax.set_title('Kernel = ' + kernel + ', C = ' + str(c))
    plt.show()
    print("Accuracies for " + kernel + " kernel: " + str(accuracies))
    print("Best accuracy with " + kernel + " kernel is " + str(top*100) + "%, calculated with C = " + str(best_c))
    # PLOT A GRAPH THAT SHOW HOW THE ACCURACY ON THE VALIDATION SET VARIES WHEN CHANGING PARAMETER C
    plt.xscale('log')
    plt.plot(C_values, accuracies)
    plt.plot(C_values, accuracies, 'bo')
    plt.xlabel('C value')
    plt.ylabel('% accuracy')
    plt.title('Accuracy with ' + kernel + ' kernel depending on C parameter')
    plt.show()

    # CALCULATE THE ACCURACY ON THE TEST SET
    result = best_clf.predict(x_test)
    accuracy = accuracy_score(y_test, result)
    print("Accuracy calculated on test set with " + kernel + " kernel is " + str(accuracy*100) + "%.")
    print()

# TUNE BOTH C AND GAMMA AT THE SAME TIME FOR RBF KERNEL
gamma_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100]
C_values = [0.1, 1, 10, 100, 1000]
param_grid = {'kernel': ['rbf'], 'C': C_values, 'gamma': gamma_values}
top = 0
fig, axes = plt.subplots(4, 2)
fig.set_size_inches(10, 14.5)
accuracies = []
for i in range(len(ParameterGrid(param_grid))):
    g = ParameterGrid(param_grid)[i]
    clf = svm.SVC(C=g['C'], kernel=g['kernel'], gamma=g['gamma'])
    clf.fit(x_train, y_train)
    result = clf.predict(x_validation)
    accuracy = accuracy_score(y_validation, result)
    accuracies.append(accuracy)
    if top < accuracy:
        top = accuracy
        best_param = g
        best_clf = clf
    ax = axes[int((i % 8) / 2)][int(i % 8) % 2]
    ax = plot_boundaries(ax, x_train, x_validation, y_validation, clf)
    ax.set_title("Kernel = " + kernel + ", C = " + str(g['C']) + ", gamma = " + str(g['gamma']))
    if i % 8 == 7:
        plt.show()
        fig, axes = plt.subplots(4, 2)
        fig.set_size_inches(10, 14.5)
        print("Accuracies for C =" + str(g['C']) + ": " + str(accuracies))
        accuracies = []


print("Best accuracy on validation set is " + str(top * 100) + "%, calculated with C = " + str(best_param['C']) +
      " and gamma = " + str(best_param['gamma']))

# CALCULATE THE ACCURACY ON THE TEST SET
result = best_clf.predict(x_test)
accuracy = accuracy_score(y_test, result)
print("Accuracy calculated on test set with RBF kernel after tuning both C and gamma is " + str(accuracy*100) + "%.")
print()

# K-FOLD CROSS VALIDATION WITH K=5
clf = GridSearchCV(svm.SVC(), param_grid, iid=True, cv=5, scoring='accuracy')
clf.fit(x_train_val, y_train_val)

print("Best parameters set found on development set: " + str(clf.best_params_))
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# CALCULATE THE ACCURACY ON THE TEST SET
result = clf.predict(x_test)
accuracy = accuracy_score(y_test, result)
print("Accuracy calculated on test set with RBF kernel after "
      "tuning the parameters with cross validation is " + str(accuracy*100) + "%.")
print()


# EXTRA: TRY DIFFERENT PAIRS OF FEATURE
# KNN
knn_param_grid = {'w': weights, 'k': ks}
svm_param_grid = {'kernel': ['rbf'], 'C': C_values, 'gamma': gamma_values}
knn_accuracy = 0
svm_accuracy = 0
for i in range(12):
    for j in range(i+1, 13):
        x_train_val, x_test, y_train_val, y_test = shuffleSplitScale(wines, i, j, 0)
        t = int((len(x_train_val) + len(x_test)) * 0.5)
        # SPLIT AGAIN TRAIN SET IN TRAIN AND VALIDATION
        x_train = x_train_val[:t, :]
        x_validation = x_train_val[t:, :]
        y_train = y_train_val[:t]
        y_validation = y_train_val[t:]
        top = 0

        # KNN
        for h in range(len(ParameterGrid(knn_param_grid))):
            g = ParameterGrid(knn_param_grid)[h]
            clf = neighbors.KNeighborsClassifier(g['k'], weights=g['w'])
            clf.fit(x_train, y_train)
            result = clf.predict(x_validation)
            accuracy = accuracy_score(y_validation, result)
            if top < accuracy:
                top = accuracy
                best_g = g
                best_clf = clf
        result = best_clf.predict(x_test)
        accuracy = accuracy_score(y_test, result)
        # print("Feature n° " + str(i+1) + " and feature N° " + str(j+1))
        # print("Best parameters set found on development set: weight = " + best_g['w'] + " and K = " + str(best_g['k']))
        # print("Accuracy calculated on test set with best parameters is " + str(accuracy*100) + "%.")
        if knn_accuracy < accuracy:
            knn_accuracy = accuracy
            knn_best_i = i
            knn_best_j = j
            knn_p = g

        # SVM
        clf = GridSearchCV(svm.SVC(), svm_param_grid, iid=True, cv=5, scoring='accuracy')
        clf.fit(x_train_val, y_train_val)
        result = clf.predict(x_test)
        accuracy = accuracy_score(y_test, result)
        #print("Feature n° " + str(i + 1) + " and feature N° " + str(j + 1))
        #print("Best parameters set found on development set: " + str(clf.best_params_))
        #print("Accuracy calculated on test set with RBF kernel after tuning the parameters with cross validation is " + str(accuracy * 100) + "%.")
        if svm_accuracy < accuracy:
            svm_accuracy = accuracy
            svm_best_i = i
            svm_best_j = j
            svm_p = clf.best_params_

print("The best pair of features for the KNN is composed of the number " + str(knn_best_i+1) + " and the number " + str(knn_best_j+1))
print("The parameters used are weight = " + knn_p['w'] + " and K = " + str(knn_p['k']))
print("The accuracy calculated on the test set is " + str(knn_accuracy*100) + "%.")
print()

print("The best pair of features for the SVM is composed of the number " + str(svm_best_i+1) + " and the number " + str(svm_best_j+1))
print("The parameters used are the RBF kernel, C = " + str(svm_p['C']) + " and gamma = " + str(svm_p['gamma']))
print("The accuracy calculated on the test set is " + str(svm_accuracy*100) + "%.")
print()
