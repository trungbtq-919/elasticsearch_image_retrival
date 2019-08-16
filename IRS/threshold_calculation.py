import json
import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy import spatial

def calculate_threshold():

    with open('evaluation_set.json', 'r') as fh:
        evaluation_set_dict = json.load(fh)
        fh.close()

    training_embedding_vectors = np.load("PCA_2048_to_512_new.npy")
    matched_dist = []
    unmatched_dist = []
    num_random = 10
    num_classes = 1000

    for i in range(num_classes):
        # print("{} : {}".format(i, evaluation_set_dict[str(i)][0]))]
        sample_vector_index = evaluation_set_dict[str(i)][0]

        if len(evaluation_set_dict[str(i)]) > 1:
            matched_vector_indices = evaluation_set_dict[str(i)][1:]
            sample_vector = training_embedding_vectors[int(sample_vector_index)]

            for matched_vector_index in matched_vector_indices:
                matched_vector = training_embedding_vectors[int(matched_vector_index)]
                # matched_dist.append(round(np.linalg.norm(sample_vector-matched_vector), 2))
                matched_dist.append(round(spatial.distance.cosine(sample_vector, matched_vector), 2))
        # print(len(matched_dist))
        data_indices = range(training_embedding_vectors.shape[0])
        unmatched_vector_indices = [item for item in data_indices if item not in evaluation_set_dict[str(i)]]
        random_unmatched_vector_indices = random.sample(unmatched_vector_indices, num_random)

        for random_unmatched_vector_index in random_unmatched_vector_indices:
            unmatched_vector = training_embedding_vectors[int(random_unmatched_vector_index)]
            # unmatched_dist.append(round(np.linalg.norm(sample_vector - unmatched_vector), 2))
            unmatched_dist.append(round(spatial.distance.cosine(sample_vector, unmatched_vector), 2))

    mean_matched_dist = np.mean(np.asarray(matched_dist))
    std_matched_dist = np.std(np.asarray(matched_dist))
    mean_unmatched_dist = np.mean(np.asarray(unmatched_dist))
    std_unmatched_dist = np.std(np.asarray(unmatched_dist))

    print(mean_matched_dist, std_matched_dist, mean_unmatched_dist, std_unmatched_dist)


    counter = collections.Counter(matched_dist)
    x = []
    y = []
    # print(counter[1])
    for i in counter.keys():
        x.append(i)
        y.append(counter[i])

    print(len(x), len(y))
    plt.hist(matched_dist, color='red', bins=100)
    plt.hist(unmatched_dist, color='blue', bins=100)
    plt.savefig("1.png")



def draw_pr_curve():

    result_1 = pd.read_csv('results/evaluation_result_euclidean_thr_1.csv', encoding='utf-8')
    result_2 = pd.read_csv('results/evaluation_result_euclidean_thr_2.csv', encoding='utf-8')
    result_3 = pd.read_csv('results/evaluation_result_euclidean_thr_3.csv', encoding='utf-8')

    with open('results/evaluation_result_euclidean_thr_1.csv', newline='') as csvfile:
        result_1 = list(csv.reader(csvfile))
        result_1 = result_1[1:]

    with open('results/evaluation_result_euclidean_thr_2.csv', newline='') as csvfile:
        result_2 = list(csv.reader(csvfile))
        result_2 = result_2[1:]

    with open('results/evaluation_result_euclidean_thr_3.csv', newline='') as csvfile:
        result_3 = list(csv.reader(csvfile))
        result_3 = result_3[1:]


    def load_precision_recall(results):
        result_pr = np.zeros((len(results), 2), dtype=float)

        for i in range(len(results)):
            precision = results[i][6]
            recall = results[i][7]

            result_pr[i, 0] += result_pr[i, 0] + round(float(precision), 4)
            result_pr[i, 1] += round(float(recall), 4)

        return result_pr


    result_pr = load_precision_recall(result_1 + result_2 + result_3)

    fig = plt.figure()
    plt.plot(result_pr[:, 1], result_pr[:, 0], marker='o', color='b')
    fig.suptitle('precision-recall curve for IRS', fontsize=20)
    plt.xlabel('recall', fontsize=18)
    plt.ylabel('precision', fontsize=16)
    fig.savefig('pr_curve.png')


def draw_hyperparams_tuning_convergence():

    with open('params_tuning_results.csv', newline='') as csvfile:
        results = list(csv.reader(csvfile))
        csvfile.close()

    # print(results)
    result_xy = np.zeros((2, len(results)))
    result_xy[0] += (np.arange(0, len(results)) + 1)

    for i in range(len(results)):
        # print(results[i])
        result_xy[1, i] += float(results[i][4])

    fig = plt.figure()
    plt.plot(result_xy[0], result_xy[1], marker='o', color='b')
    fig.suptitle('params tuning convergence using PSO', fontsize=20)
    plt.xlabel('iteration', fontsize=14)
    plt.ylabel('score = 2 - precision - recall', fontsize=12)
    fig.savefig('params_tuning_curve.png')


calculate_threshold()