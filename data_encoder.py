import numpy as np
from sklearn.cluster import KMeans
import csv
import os
import pickle

class DataEncoder:

    def __init__(self):

        self.num_groups = 16
        self.num_clusters = 20


    def preprocess(self):
        """ subdivide embedding vector into groups for clustering
        """

        dimension = self.embs_vector.shape[1]
        group_dimension = int(dimension / self.num_groups)
        preprocessed_embs_vector = []
       
        for group_id in range(self.num_groups):
            a = group_id * group_dimension
            b = (group_id + 1) * group_dimension

            group_i = self.embs_vector[0, a:b]
            preprocessed_embs_vector.append(group_i)

        return preprocessed_embs_vector


    def encode(self, embs_vector):
        """ Genterate encoded string tokens
        """

        self.embs_vector = embs_vector

        preprocessed_embs_vector = self.preprocess()

        # Load kmeans model for encoding.
        kmeans = []
        kmeans_model_path = "model/kmeans_models_group_"
        for group in range(self.num_groups):
            with open(kmeans_model_path + str(group + 1) + '.pckl', "rb") as f:
                kmeans.append(pickle.load(f))

        ############## Clustering ##############
        # Predict cluster for each subvector
        subvector_clusters = []
        for i in range(self.num_groups):
            group = preprocessed_embs_vector[i]
            label = kmeans[i].predict(X=group.reshape((1, -1)))
            subvector_clusters.append(label)

        ############# Tokenize ################
        string_tokens = []
        for idx in range(self.num_groups):
            string_tokens.append( "position" + str(idx + 1) + "cluster" + str(subvector_clusters[idx][0]) )

        return string_tokens




#         kmeans = []
#         num_data_points = self.embs_vector.shape[0]
#         labels = np.zeros((num_data_points, self.num_groups))

#         directory = './encode_results/encode_' + str(self.num_groups) + 'groups_' \
#                          + str(self.num_clusters) + 'clusters'
#         kmeans_model_name = 'kmeans_models.pckl'
#         kmeans_models_path = directory + '/' + kmeans_model_name

#         if not os.path.exists(kmeans_models_path):
#             for i in range(len(preprocessed_train_embs)):
#                 group_i = preprocessed_train_embs[i]
#                 kmeans_i = KMeans(n_clusters=self.num_clusters, n_init=10, max_iter=self.k_means_iter)
#                 label_i = kmeans_i.fit_predict(X=group_i)
#                 kmeans.append(kmeans_i)
#                 labels[:, i] += label_i

#         encoded_string_list = []
#         for i in range(num_data_points):
#             encoded_string_i = []
#             for j in range(self.num_groups):
#                 encoded_string_ij = 'position' + str(int(j + 1)) + 'cluster' + str(int(labels[i, j]))
#                 encoded_string_i.append(encoded_string_ij)

#             encoded_string_list.append(encoded_string_i)

#         return kmeans, encoded_string_list

#     def run_encode_data(self):

#         print('{} groups, {} clusters'.format(self.num_groups, self.num_clusters))

#         preprocessed_train_embs = self.preprocessing()
#         kmeans, encoded_string_list = self.encode_string_tokens(preprocessed_train_embs)
#         self.save_results(kmeans, encoded_string_list)


# def main():
#     # Path to embedding vectors
#     face_embedding_vectors_path = "hyperparams_training_data/train_embs.npy"
#     train_embs = np.load(train_embs)

#     nums_groups = [8, 16, 32, 64]
#     nums_clusters = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#                      31, 32]

#     # nums_groups = [8]
#     # nums_clusters = [8]
#     max_iter = 1000

#     for num_groups in nums_groups:
#         for num_clusters in nums_clusters:
#             data_encoder = DataEncoder(num_groups, num_clusters, max_iter, train_embs[0].reshape(1, train_embs.shape[1]))
#             data_encoder.run_encode_data()


# if __name__ == '__main__':
#     main()
