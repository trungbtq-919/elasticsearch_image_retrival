""" file_name: hyper_tuning.py

- Hyperparameters tuning for best choosing K (num_clusters)
and number of subvectors (num_groups).

"""


import DataEncoder

def hyper_tuning(num_groups: int, num_clusters: int, max_iters, validation_data):
	
	encoder = DataEncoder(num_groups, num_clusters, max_iters, validation_data)



def save_results(kmeans, encoded_string_list):

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

def main():

	# Number of subvectors.
	num_groups = [8, 16, 32, 64]

	# hyper-params K for K-Means algorithm.
	lower = 8
	upper = 32
	num_clusters = [cluster for cluster in range(lower, upper + 1)]

	max_iters = 1000

	for num_group in num_groups:
		for num_cluster in num_clusters:
			hyper_tuning(num_group, num_cluster, )


if __name__ == "__main__":
	main()