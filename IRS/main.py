from  index_data import ESIndexer
from search import Searcher
from data_encoder import DataEncoder
from elasticsearch import Elasticsearch
from json_generator import JsonStringTokenGenerator
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

    training_embedding_vectors = np.load("train_embs_VGGFace.npy")
    query_vector_indices = random.sample(range(len(evaluation_set.keys())), num_queries)
    train_labels, image_names = get_image_data('vn_celeb_face_recognition/train.csv')

    for threshold in thresholds:
        for num_groups in nums_groups:
            for num_clusters in nums_clusters:

                print("working on {} groups, {} clusters, {} threshold".format(num_groups, num_clusters, threshold))
                search_times = []
                mean_average_accuracy = 0
                mean_recall = 0
                for query_vector_index in query_vector_indices:

                    query_vector = training_embedding_vectors[evaluation_set[str(query_vector_index)][0]]
                    actual_query_label = train_labels[evaluation_set[str(query_vector_index)][0]]
                    num_actual_results = len(evaluation_set[str(actual_query_label)])
                    # print(actual_query_label)
                    # print("------------")

                    es = Elasticsearch(server_url)
                    index_name = 'face_off_' + str(num_groups) + 'groups_' + str(num_clusters) + 'clusters_vgg'
                    if not es.indices.exists(index_name): # if data is not indexed, create index and take data to ES
                                                        # then query
                        indexer = ESIndexer('encode_results_vgg', num_groups, num_clusters, server_url, 'vgg')
                        indexer.index()

                        start_time = datetime.now()
                        searcher = Searcher(threshold, num_groups, num_clusters, query_vector, server_url, index_name,
                                            'cosine', 'vgg')
                        results = searcher.search()
                        # print(len(results))
                        if len(results) == 0: continue
                        search_time = datetime.now() - start_time
                        search_time_in_ms = (search_time.days * 24 * 60 * 60 +
                                             search_time.seconds) * 1000 + \
                                             search_time.microseconds / 1000.0
                        search_times.append(search_time_in_ms)
                    else: # if not, commit query
                        start_time = datetime.now()
                        searcher = Searcher(threshold, num_groups, num_clusters, query_vector, server_url, index_name,
                                            'cosine', 'vgg')
                        results = searcher.search()
                        # print(len(results))
                        if len(results) == 0: continue
                        search_time = datetime.now() - start_time
                        search_time_in_ms = (search_time.days * 24 * 60 * 60 +
                                             search_time.seconds) * 1000 + \
                                            search_time.microseconds / 1000.0
                        search_times.append(search_time_in_ms)

                    # print(len(results))
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


def main_1(var):

    num_groups = int(var[0])

    num_clusters = int(var[1])

    if var[2] >= 50:
        dist_function_name = 'euclidean'
    else:
        dist_function_name = 'cosine'
    threshold = var[3]

    server_url = 'localhost:9200'
    num_queries = 200

    with open('evaluation_set.json') as f:
        evaluation_set = json.load(f)
        f.close()

    training_embedding_vectors = np.load("PCA_2048_to_512_new.npy")
    query_vector_indices = random.sample(range(len(evaluation_set.keys())), num_queries)
    train_labels, image_names = get_image_data('vn_celeb_face_recognition/train.csv')

    # print("working on {} groups, {} clusters, {} threshold".format(num_groups, num_clusters, threshold))
    search_times = []
    mean_average_accuracy = 0
    mean_recall = 0

    for query_vector_index in query_vector_indices:

        query_vector = training_embedding_vectors[evaluation_set[str(query_vector_index)][0]]
        # print(query_vector)
        actual_query_label = train_labels[evaluation_set[str(query_vector_index)][0]]
        num_actual_results = len(evaluation_set[str(actual_query_label)])
        # print(actual_query_label)
        # print("------------")

        es = Elasticsearch(server_url)
        index_name = 'face_off_' + str(num_groups) + 'groups_' + str(num_clusters) + 'clusters_vgg'
        if not es.indices.exists(index_name):  # if data is not indexed, create index and take data to ES
            # then query
            data_encoder = DataEncoder(num_groups, num_clusters, 1000, training_embedding_vectors, 'encode_results_vgg')
            data_encoder.run_encode_data()
            json_string_tokens_generator = JsonStringTokenGenerator('encode_results_vgg', 'PCA_2048_to_512_new.npy',
                                                                    'vn_celeb_face_recognition/train.csv', num_groups,
                                                                    num_clusters)
            encoded_string_tokens_list = json_string_tokens_generator.get_string_tokens_list()
            train_embs = json_string_tokens_generator.get_image_fetures()
            train_labels, image_names = json_string_tokens_generator.get_image_metadata()
            json_string_tokens_list = json_string_tokens_generator.generate_json_string_tokens_list(
                encoded_string_tokens_list,
                train_labels,
                image_names,
                train_embs)
            json_string_tokens_generator.save_json_string_tokens(json_string_tokens_list)

            print('saving completed....')
            print('******************************')
            indexer = ESIndexer('encode_results_vgg', num_groups, num_clusters, server_url, 'vgg')
            indexer.index()

            start_time = datetime.now()
            searcher = Searcher(threshold, num_groups, num_clusters, query_vector, server_url, index_name,
                                dist_function_name, 'vgg')
            results = searcher.search()
            # print(len(results))
            if len(results) == 0: continue
            search_time = datetime.now() - start_time
            search_time_in_ms = (search_time.days * 24 * 60 * 60 +
                                 search_time.seconds) * 1000 + \
                                search_time.microseconds / 1000.0
            search_times.append(search_time_in_ms)
        else:  # if not, commit query
            start_time = datetime.now()
            searcher = Searcher(threshold, num_groups, num_clusters, query_vector, server_url, index_name,
                                dist_function_name, 'vgg')
            results = searcher.search()
            # print(len(results))
            if len(results) == 0: continue
            search_time = datetime.now() - start_time
            search_time_in_ms = (search_time.days * 24 * 60 * 60 +
                                 search_time.seconds) * 1000 + \
                                search_time.microseconds / 1000.0
            search_times.append(search_time_in_ms)

        results_labels = list()
        for result in results:
            results_labels.append(result['id'])

        # with open('evaluation_set.json', 'r') as fh:
        #     evaluation_set_dict = json.load(fh)
        #     fh.close()

        accuracy_i = 0
        for i in range(len(results)):
            step_list = results_labels[:(i + 1)]
            num_corrects = len([i for i, x in enumerate(step_list) if x == actual_query_label])
            accuracy_i += num_corrects / len(step_list)
        # print(accuracy_i/num_returns)
        mean_average_accuracy += accuracy_i / len(results)

        recall_i = num_corrects / num_actual_results
        # print(num_corrects)
        mean_recall += recall_i

        # print("*************************************")

    mean_average_accuracy = mean_average_accuracy / num_queries
    mean_recall = mean_recall / num_queries
    print(mean_average_accuracy, mean_recall)
    # print("precision: {} and recall: {}".format(mean_average_accuracy, mean_recall))
    # print(average_search_time)
    # print(mean_average_accuracy)

    return 3 - mean_average_accuracy - mean_recall - (2*mean_average_accuracy*mean_recall
                                                      / (mean_average_accuracy + mean_recall))

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