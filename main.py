from index_data import ESIndexer
from search import Searcher
from elasticsearch import Elasticsearch
import numpy as np
import csv
import json
import random
from datetime import datetime
import os.path


def get_image_data(image_id_path):

    file = open(image_id_path, 'r')
    reader = csv.reader(file)
    train_labels = []
    image_names = []
    for line in reader:
        image_names.append(line[0])
        train_labels.append(line[1])

    train_labels = train_labels[1:]
    image_names = image_names[1:]

    return train_labels, image_names


def main():
    server_url = 'localhost:9200'
    num_queries = 1000

    with open('hyper_params_set.json', 'r') as fh:
        hyper_params = json.load(fh)
        nums_groups = hyper_params['nums_groups']
        nums_clusters = hyper_params['nums_clusters']
        thresholds = hyper_params['thresholds']
        fh.close()

    with open('evaluation_set.json') as f:
        evaluation_set = json.load(f)
        f.close()

    final_results = []

    training_embedding_vectors = np.load("train_embs.npy")
    query_vector_indices = random.sample(range(training_embedding_vectors.shape[0]), num_queries)
    train_labels, image_names = get_image_data('vn_celeb_face_recognition/train.csv')

    for threshold in thresholds:
        for num_groups in nums_groups:
            for num_clusters in nums_clusters:

                print("working on {} groups, {} clusters, {} threshold".format(num_groups, num_clusters, threshold))
                search_times = []
                mean_average_accuracy = 0
                mean_recall = 0
                for query_vector_index in query_vector_indices:

                    query_vector = training_embedding_vectors[query_vector_index]
                    actual_query_label = train_labels[query_vector_index]
                    num_actual_results = len(evaluation_set[str(actual_query_label)])
                    # print(actual_query_label)
                    # print("------------")

                    es = Elasticsearch(server_url)
                    index_name = 'face_off_' + str(num_groups) + 'groups_' + str(num_clusters) + 'clusters'
                    if not es.indices.exists(index_name):
                        indexer = ESIndexer(num_groups, num_clusters, server_url)
                        indexer.index()

                        start_time = datetime.now()
                        searcher = Searcher(threshold, num_groups, num_clusters, query_vector, server_url, index_name)
                        results = searcher.search()
                        if len(results) == 0: continue
                        search_time = datetime.now() - start_time
                        search_time_in_ms = (search_time.days * 24 * 60 * 60 +
                                             search_time.seconds) * 1000 + \
                                             search_time.microseconds / 1000.0
                        search_times.append(search_time_in_ms)
                    else:
                        start_time = datetime.now()
                        searcher = Searcher(threshold, num_groups, num_clusters, query_vector, server_url, index_name)
                        results = searcher.search()
                        if len(results) == 0: continue
                        search_time = datetime.now() - start_time
                        search_time_in_ms = (search_time.days * 24 * 60 * 60 +
                                             search_time.seconds) * 1000 + \
                                            search_time.microseconds / 1000.0
                        search_times.append(search_time_in_ms)

                    results_labels = list()
                    for result in results:
                        # print(result['id'])
                        results_labels.append(result['id'])

                    # with open('evaluation_set.json', 'r') as fh:
                    #     evaluation_set_dict = json.load(fh)
                    #     fh.close()

                    accuracy_i = 0
                    for i in range(len(results)):
                        step_list = results_labels[:(i+1)]
                        num_corrects = len([i for i, x in enumerate(step_list) if x == actual_query_label])
                        accuracy_i += num_corrects/len(step_list)
                    # print(accuracy_i/num_returns)
                    mean_average_accuracy += accuracy_i/len(results)

                    recall_i = num_corrects/num_actual_results
                    # print(num_corrects)
                    mean_recall += recall_i

                    # print("*************************************")
                average_search_time = round(np.mean(np.asarray(search_times))/1000, 3)
                mean_average_accuracy = mean_average_accuracy / num_queries
                mean_recall = mean_recall/num_queries
                # print(average_search_time)
                # print(accuracy)

                final_results.append([num_groups, num_clusters, threshold, num_queries, 'euclidean', average_search_time,
                                      round(mean_average_accuracy, 4), round(mean_recall, 4)])
                print([num_groups, num_clusters, threshold, num_queries, 'euclidean', average_search_time,
                                      round(mean_average_accuracy, 4), round(mean_recall, 4)])

                print("finish")
                print("-----------------------------------------------")

    # results_path = "./results/evaluation_result_euclidean_thr_1.csv"
    # if not os.path.exists(results_path):
    #     with open(results_path, 'w') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['num_groups', 'num_clusters', 'threshold', 'num_queries', 're_rank',
    #                          'average_search_time', 'mean_average_accuracy', 'mean_recall'])
    #         writer.writerows(final_results)



    # train_labels, image_names = get_image_data('vn_celeb_face_recognition/train.csv')
    # evaluation_set = dict()
    #
    # for train_label in train_labels:
    #     indexes = [i for i, x in enumerate(train_labels) if x == train_label]
    #     evaluation_set[str(train_label)] = indexes
    #
    # with open('evaluation_set.json', 'w') as fp:
    #     json.dump(evaluation_set, fp)


if __name__ == '__main__':
    main()