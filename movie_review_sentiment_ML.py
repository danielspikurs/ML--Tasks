import pandas as pd
import numpy as np
import re
import math
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics

# Machine Learning Assignment 1
# Daniels Pikurs

data = pd.read_excel("movie_reviews.xlsx")
data["Sentiment"] = data["Sentiment"].map({"negative": 0, "positive": 1})


def task_one(data):
    train_data = data[data["Split"]=="train"][["Review"]].to_numpy()
    test_data = data[data["Split"]=="test"][["Review"]].to_numpy()
    
    train_labels = data[data["Split"]=="train"][["Sentiment"]].to_numpy()
    test_labels = data[data["Split"]=="test"][["Sentiment"]].to_numpy()

    pos_reviews_train = len(data[(data["Sentiment"]==1) & (data["Split"]=="train")])
    neg_reviews_train = len(data[(data["Sentiment"]==0) & (data["Split"]=="train")])

    pos_reviews_test = len(data[(data["Sentiment"]==1) & (data["Split"]=="test")])
    neg_reviews_test = len(data[(data["Sentiment"]==0) & (data["Split"]=="test")])
    
    print("\nNumber of Positive Reviews in Training Set: ", pos_reviews_train)
    print("Number of Negative Reviews in Training Set: ", neg_reviews_train)
    print("")
    print("Number of Positive Reviews in Evaluation Set: ", pos_reviews_test)
    print("Number of Negative Reviews in Evaluation Set: ", neg_reviews_test)

    total_negative_reviews = neg_reviews_train + neg_reviews_test
    total_positive_reviews = pos_reviews_train + pos_reviews_test

    return train_data, train_labels, test_data, test_labels, total_positive_reviews, total_negative_reviews


def task_two(train_data, min_word_len, min_word_occ):
    word_occurences = {}
    filtered_words = []
    
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            train_data[i][j] = re.sub(r'[^a-zA-Z]', ' ', train_data[i][j])
            train_data[i][j] = re.sub(' +', ' ', train_data[i][j])
            train_data[i][j] = train_data[i][j].lower()
            
            all_words = train_data[i][j].split()
            for word in all_words:
                if (len(word) >= min_word_len):
                    if (word in word_occurences):
                        word_occurences[word] = word_occurences[word] + 1
                    else:
                        word_occurences[word] = 1
    
    for word in word_occurences:
        if word_occurences[word] >= min_word_occ:
            filtered_words.append(word)
    
    return filtered_words
    
    
def task_three(train_data, words_to_filter, train_labels):
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            review_words = train_data[i][j].split()
            words_to_keep = []
            for word in review_words:
                if word in words_to_filter:
                     words_to_keep.append(word)
            train_data[i][j] = " ".join(words_to_keep)

    positive_reviews = train_data[train_labels == 1]
    negative_reviews = train_data[train_labels == 0]
    
    positive_dictionary = {words_to_filter[i]: 0 for i in range(0, len(words_to_filter))}
    negative_dictionary = {words_to_filter[i]: 0 for i in range(0, len(words_to_filter))}
    
    for i in range(len(positive_reviews)):  
        review_words = positive_reviews[i].split()
        for word in positive_dictionary:
            if word in review_words:
                positive_dictionary[word] = positive_dictionary[word] + 1

    for i in range(len(negative_reviews)):  
        review_words = negative_reviews[i].split()
        for word in negative_dictionary:
            if word in review_words:
                negative_dictionary[word] = negative_dictionary[word] + 1

    return positive_dictionary, negative_dictionary


def task_four(total_pos_rev, total_neg_rev, pos_feat_count, neg_feat_count, relevant_features):
    total_reviews = total_pos_rev + total_neg_rev

    probability_pos_rev = total_pos_rev / total_reviews
    probability_neg_rev = total_neg_rev / total_reviews

    pos_word_dict = {}
    neg_word_dict = {}

    word_count_positive_rev = 0
    word_count_negative_rev = 0

    # Counting Words/Features in Positive/Negative Reviews
    for feature in pos_feat_count:
        word_count_positive_rev = word_count_positive_rev + pos_feat_count[feature]
    for feature in neg_feat_count:
        word_count_negative_rev = word_count_negative_rev + neg_feat_count[feature]

    smoothing_factor = 1

    for word in relevant_features:
        pos_word_dict[word] = (pos_feat_count[word] + smoothing_factor) / (word_count_positive_rev + 2 * smoothing_factor)
        neg_word_dict[word] = (neg_feat_count[word] + smoothing_factor) / (word_count_negative_rev + 2 * smoothing_factor)

    return pos_word_dict, neg_word_dict, probability_pos_rev, probability_neg_rev
    

def task_five(my_review, prob_pos_prior, prob_neg_prior, positive_feature_likelihood, negative_feature_likelihood):
    prediction = 0

    my_review_words = my_review.split()

    log_likelihood_positive = 0
    log_likelihood_negative = 0

    for word in my_review_words:
        if positive_feature_likelihood.get(word) is not None:
            log_likelihood_positive = log_likelihood_positive + math.log(positive_feature_likelihood.get(word))
            log_likelihood_negative = log_likelihood_negative + math.log(negative_feature_likelihood.get(word))

    if log_likelihood_positive - log_likelihood_negative > math.log(prob_neg_prior) - math.log(prob_pos_prior):
        prediction = 1
    else:
        prediction = 0
    
    return prediction


def task_six(train_data, train_labels, test_labels, test_data):
    positive_reviews = train_data[train_labels == 1]
    negative_reviews = train_data[train_labels == 0]

    accuracy_score = []

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    kf = model_selection.StratifiedKFold(n_splits = min(positive_reviews, negative_reviews), shuffle=True)
    
    for train_index, test_index in kf.split(train_data, test_labels):
        clf = svm.SVC()
        clf.fit(train_data[train_index, :], test_labels[train_index])
        predicted_labels = clf.predict(train_data[test_index, :])

        c = metrics.confusion_matrix(test_labels[test_index], predicted_labels)

        true_positives.append(c[0,0])
        true_negatives.append(c[1,1])
        false_positives.append(c[1,0])
        false_negatives.append(c[0,1])

    print("True Positives: ", true_positives)
    print("True Negatives: ", true_negatives)
    print("False Positives: ", false_positives)
    print("False Negatives: ", false_negatives)


def main():
    min_word_length = 6
    min_word_occurence = 500
    
    # Task 1 Splitting and Counting Reviews
    train_data, train_labels, test_data, test_labels, total_positive_reviews, total_negative_reviews = task_one(data)
    
    # Task 2 Extracting Relevant Features
    relevant_features = task_two(train_data, min_word_length, min_word_occurence)
    
    # Task 3 Counting Feature Fequencies
    positive_feature_count, negative_feature_count = task_three(train_data, relevant_features, train_labels)
    
    # Task 4 Calculate Leature Likelihoods and Priors
    positive_feature_likelihood, negative_feature_likelihood, prob_pos_prior, prob_neg_prior = task_four(total_positive_reviews, total_negative_reviews, positive_feature_count, negative_feature_count, relevant_features)

    #my_review = "this movie is amazing i loved it so much, the actors were great, the performance was amazing"
    my_review = "this move was terrible, i hated it so much, actors were bad, i was not satisfied"

    # Task 5 Maximum Likelihood Classification
    prediction = task_five(my_review, prob_pos_prior, prob_neg_prior, positive_feature_likelihood, negative_feature_likelihood)
    
    if prediction == 1:
        print("\nThe prediction for your review is Positive")
    else:
        print("\nThe prediction for your review is Negative")

    # Task 6 Evaluation of Results
    task_six(train_data, train_labels)

main()
