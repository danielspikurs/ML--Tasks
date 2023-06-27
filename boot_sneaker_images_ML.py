import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors
from matplotlib import pyplot as plt

# Machine Learning Assignment 2
# Daniels Pikurs R00166279
# SDH4-B 

# 0 Sneaker, 1 Boot

data = pd.read_csv("product_images.csv")
accuracy_ranking = {}
time_ranking = {}

def task_one():
    feature_vectors = np.array(data.iloc[:,1:])
    labels = np.array(data.iloc[:,0])

    s_count = len(data[(data["label"]==0)])
    b_count = len(data[(data["label"]==1)])

    print("Sneaker samples: ", s_count)
    print("Boot samples: ", b_count)

    s_index = 3
    b_index = 2
    sample_boot = feature_vectors[b_index,:].reshape(28,28)
    sample_sneaker = feature_vectors[s_index,:].reshape(28,28)

    plt.imshow(sample_boot)
    plt.axis('off')
    plt.show()

    plt.imshow(sample_sneaker)
    plt.axis('off')
    plt.show()

    return feature_vectors, labels, s_count, b_count


def task_two(fv, labels, classifier, nsplits, nrows=5000):
    training_times = []
    prediction_times = []
    prediction_accuracies = []

    classifier_runtime_start = time.perf_counter()
    print("\nTRAINING CLASSIFIER \nUsing NSplits: ", nsplits, "\nUsing Classifier: ", classifier)
    kf = model_selection.KFold(n_splits = nsplits, shuffle=True)


    for train_index, test_index in kf.split(fv[:nrows], labels[:nrows]):
        train_time_before = time.perf_counter()
        classifier.fit(fv[train_index], labels[train_index])
        train_time_after = time.perf_counter()

        training_time = train_time_after - train_time_before
        training_times.append(training_time)

        print("\nTraining Time: ", training_time)

        predict_time_before = time.perf_counter()
        predicted_labels = classifier.predict(fv[test_index, :])
        predict_time_after = time.perf_counter()

        prediction_time = predict_time_after - predict_time_before
        prediction_times.append(prediction_time)

        print("Predicting Time: ", prediction_time)
            
        c = metrics.confusion_matrix(labels[test_index], predicted_labels)

        accuracy = metrics.accuracy_score(labels[test_index], predicted_labels)
        prediction_accuracies.append(accuracy)

        print("Prediction Accuracy: ", accuracy)
        print("\nConfusion Matrix\n", c)
        print("\n-------------------------------------------------")
    classifier_runtime_end = time.perf_counter()

    print("\nFold: ", nsplits," Classifier: ", classifier)
    print("\nTraining Time Average: ", np.average(training_times))
    print("Max Training Time: ", np.max(training_times))
    print("Min Training Time: ", np.min(training_times))

    print("\nPrediction Time Average: ", np.average(prediction_times))
    print("Max Prediction Time: ", np.max(prediction_times))
    print("Min Prediction Time: ", np.min(prediction_times))

    print("\nPrediction Accuracy Average: ", np.average(prediction_accuracies))
    print("Max Prediction Accuracy: ", np.max(prediction_accuracies))
    print("Min Prediction Accuracy: ", np.min(prediction_accuracies))

    accuracy_ranking[classifier] = np.max(prediction_accuracies)
    time_ranking[classifier] = classifier_runtime_end - classifier_runtime_start

    print("\n-------------------------------------------------")

    classifier_runtime = classifier_runtime_end - classifier_runtime_start
    return prediction_accuracies, classifier_runtime



def task_three(fv, labels):
    print("\nTask three")
    clf = linear_model.Perceptron()
    runtimes = []
    mean_accuracies = []
    for nrows in [1000, 5000, 10000]:
        perceptron_prediction_accuracies, classifier_runtime = task_two(fv, labels, clf, 2, nrows)
        print("\nMean Prediction Accuracy for Perceptron is: ", np.mean(perceptron_prediction_accuracies))
        print("\n-------------------------------------------------")
        runtimes.append(classifier_runtime)
        mean_accuracies.append(np.mean(perceptron_prediction_accuracies))


    plt.plot([1000, 5000, 10000], runtimes)
    plt.xlabel('Sample Size')
    plt.ylabel('Runtime')
    plt.title('Perceptron Sample Size vs Runtimes')
    plt.show()

    
def task_four(fv, labels):
    folds = 2
    mean_accuracies = []
    runtimes = []
    for g in (0.001, 0.1, 1):
        clf = svm.SVC(kernel='rbf', gamma = g)
        svm_prediction_accuracies, classifier_runtime = task_two(fv, labels, clf, folds)
        print("Using 𝛾 of: ", g)
        print("\nMean Prediction Accuracy for SVM is: ", np.mean(svm_prediction_accuracies))
        mean_accuracies.append(np.mean(svm_prediction_accuracies))
        runtimes.append(classifier_runtime)


    plt.plot([0.001, 0.1, 1], mean_accuracies)
    plt.xlabel('Gamma Value')
    plt.ylabel('Accuracies')
    plt.title('SVM Gamma vs Accuracy')
    plt.show() 

    # From the graph I can see that the gamma value of 1 provides the most accurate predictions.
    # The best mean prediction score is just about 50%.

    runtimes = []
    for nrows in [1000, 5000, 10000]:
        clf = svm.SVC(kernel='rbf', gamma = 1)
        svm_prediction_accuracies, classifier_runtime = task_two(fv, labels, clf, folds, nrows)
        print("Using 𝛾 of: ", g)
        print("\nMean Prediction Accuracy for SVM is: ", np.mean(svm_prediction_accuracies))
        mean_accuracies.append(np.mean(svm_prediction_accuracies))
        runtimes.append(classifier_runtime)

    plt.plot([1000, 5000, 10000], runtimes)
    plt.xlabel('Sample Size')
    plt.ylabel('Runtimes')
    plt.title('SVM Sample Size vs Runtimes')
    plt.show() 


def task_five(fv, labels):
    folds = 2
    mean_accuracies = []
    runtimes = []
    for k in [2, 4, 6]:
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn_prediction_accuracies, classifier_runtime =  task_two(fv, labels, clf, folds)
        print("Using K value of: ", k)
        print("\nMean Prediction Accuracy for KNN is: ", np.mean(knn_prediction_accuracies))
        mean_accuracies.append(np.mean(knn_prediction_accuracies))
        runtimes.append(classifier_runtime)

    plt.plot([2, 4, 6], mean_accuracies)
    plt.xlabel('K Values')
    plt.ylabel('Accuracies')
    plt.title('KNN K vs Accuracy')
    plt.show() 

    # From the Graph K of 2 provided the best accuracy.
    # Best achievable accuracy is almost 95%.
    # Mean prediction accuracy for KNN is 93%

    runtimes = []
    for nrows in [1000, 5000, 10000]:
        clf = neighbors.KNeighborsClassifier(n_neighbors=2)
        knn_prediction_accuracies, classifier_runtime =  task_two(fv, labels, clf, folds, nrows)
        print("Using K value of: ", k)
        print("\nMean Prediction Accuracy for KNN is: ", np.mean(knn_prediction_accuracies))
        mean_accuracies.append(np.mean(knn_prediction_accuracies))
        runtimes.append(classifier_runtime)

    plt.plot([1000, 5000, 10000], runtimes)
    plt.xlabel('Sample Size')
    plt.ylabel('Runtimes')
    plt.title('KNN Sample Size vs Runtimes')
    plt.show() 

def task_six(fv, labels):
    runtimes = []
    for nrows in ([1000, 5000, 10000]):
        clf = tree.DecisionTreeClassifier()
        dt_prediction_accuracies, classifier_runtime = task_two(fv, labels, clf, 2, nrows)
        print("\nMean Prediction Accuracy for Decision Tree is: ", np.mean(dt_prediction_accuracies))
        runtimes.append(classifier_runtime)

    plt.plot([1000, 5000, 10000], runtimes)
    plt.xlabel('Sample Size')
    plt.ylabel('Runtimes')
    plt.title('DecisionTree Sample Size vs Runtimes')

    # Mean prediction accuracy of Decision tree is 89%


def task_seven():
    # Answer in PDF
    print("\n\nAnswer to Task 7 in PDF")
    print("Accuracy Rankings of Classifiers\n", accuracy_ranking)
    print("\nTime Rankings of Classifiers\n", time_ranking)


feature_vectors, labels, sneaker_count, boot_count = task_one()
task_three(feature_vectors, labels)
task_four(feature_vectors, labels)
task_five(feature_vectors, labels)
task_six(feature_vectors, labels)
task_seven()