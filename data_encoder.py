import numpy as np
from sklearn.cluster import KMeans
import csv
import os
import pickle


class DataEncoder(object):

    def __init__(self, num_groups, num_clusters, k_means_iter, train_embs):
        self.num_groups = num_groups
        self.num_clusters = num_clusters
        self.k_means_iter = k_means_iter
        self.train_embs = train_embs

    def preprocess_train_embs_vectors(self):  # divide embs vectors into groups for clustering

        dimension = self.train_embs.shape[1]

        group_dimension = int(dimension / self.num_groups)
        preprocessed_train_embs = []
        for group_id in range(self.num_groups):
            a = group_id * group_dimension
            b = (group_id + 1) * group_dimension

            group_i = self.train_embs[:, a:b]
            preprocessed_train_embs.append(group_i)

        return preprocessed_train_embs

    def encode_string_tokens(self):  # genterate encoded string tokens

        preprocessed_train_embs = self.preprocess_train_embs_vectors()

        kmeans = []
        num_data_points = self.train_embs.shape[0]
        labels = np.zeros((num_data_points, self.num_groups))

        directory = './encode_results/encode_' + str(self.num_groups) + 'groups_' \
                         + str(self.num_clusters) + 'clusters'
        kmeans_model_name = 'kmeans_models.pckl'
        kmeans_models_path = directory + '/' + kmeans_model_name

        if not os.path.exists(kmeans_models_path):
            for i in range(len(preprocessed_train_embs)):
                group_i = preprocessed_train_embs[i]
                kmeans_i = KMeans(n_clusters=self.num_clusters, n_init=10, max_iter=self.k_means_iter)
                label_i = kmeans_i.fit_predict(X=group_i)
                kmeans.append(kmeans_i)
                labels[:, i] += label_i
        else:
            with open(kmeans_models_path, "rb") as f:
                while True:
                    try:
                        kmeans.append(pickle.load(f))
                    except EOFError:
                        break

            for i in range(len(preprocessed_train_embs)):
                group_i = preprocessed_train_embs[i]
                label_i = kmeans[i].predict(X=group_i)
                labels[:, i] += label_i

        # print(len(kmeans), labels.shape)

        encoded_string_list = []
        for i in range(num_data_points):
            encoded_string_i = []
            for j in range(self.num_groups):
                encoded_string_ij = 'position' + str(int(j + 1)) + 'cluster' + str(int(labels[i, j]))
                encoded_string_i.append(encoded_string_ij)

            encoded_string_list.append(encoded_string_i)

        return kmeans, encoded_string_list

    def save_results(self, kmeans, encoded_string_list): #### SAVE KMEANS MODEL AND
                                                            # ENCODED STRING TOKENS #######

        csv_file_name = 'encoded_string_' + str(self.num_groups) + 'groups_' + str(self.num_clusters) + 'clusters.csv'

        directory_name = './encode_results/encode_' + str(self.num_groups) + 'groups_' \
                         + str(self.num_clusters) + 'clusters'

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        csv_file_name_path = directory_name + '/' + csv_file_name

        if not os.path.exists(csv_file_name_path):
            with open(directory_name + '/' + csv_file_name, "w",
                      newline="") as f:
                writer = csv.writer(f)
                writer.writerows(encoded_string_list)

        kmeans_model_name = 'kmeans_models.pckl'
        kmeans_model_name_path = directory_name + '/' + kmeans_model_name

        if not os.path.exists(kmeans_model_name_path):
            with open(kmeans_model_name_path, "wb") as f:
                for kmean in kmeans:
                    pickle.dump(kmean, f)

    def run_encode_data(self):

        print('start to encode with {} groups and {} clusters'.format(self.num_groups, self.num_clusters))

        preprocessed_train_embs = self.preprocess_train_embs_vectors()
        kmeans, encoded_string_list = self.encode_string_tokens(preprocessed_train_embs)
        self.save_results(kmeans, encoded_string_list)


def encode_main():
    train_embs = np.load('train_embs.npy')
    # print(train_embs.shape)

    nums_groups = [8, 16, 32, 64]
    nums_clusters = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                     31, 32]

    # nums_groups = [8]
    # nums_clusters = [8]
    max_iter = 1000

    for num_groups in nums_groups:
        for num_clusters in nums_clusters:
            data_encoder = DataEncoder(num_groups, num_clusters, max_iter, train_embs[0].reshape(1, train_embs.shape[1]))
            data_encoder.run_encode_data()


if __name__ == '__main__':
    encode_main()



# print(encoded_string_list[0])
# print(encoded_string_list[512])
# print(encoded_string_list[946])
# print(encoded_string_list[2650])
# print(encoded_string_list[3007])
# print(encoded_string_list[3928])
# print(encoded_string_list[4600])


