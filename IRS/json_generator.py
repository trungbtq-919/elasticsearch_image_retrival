import os
import json
import numpy as np
import csv

path = './encode_results'
features_csv_path = './train_embs.npy'


class JsonStringTokenGenerator(object):

    def __init__(self, encode_string_tokens_directory, train_embs_directory, image_metadata_directory,
                 num_groups, num_clusters):
        self.encode_string_tokens_directory = encode_string_tokens_directory + '/encode_' + str(num_groups) + \
                                              "groups_" + str(num_clusters) + 'clusters'
        self.train_embs_directory = train_embs_directory
        self.image_metadata_directory = image_metadata_directory
        self.num_groups = num_groups
        self.num_clusters = num_clusters

    def get_string_tokens_list(self):  ## get encoded string tokens from csv files ##
        string_tokens_path = os.path.join(self.encode_string_tokens_directory,
                                          'encoded_string_' + str(self.num_groups) + 'groups_'
                                          + str(self.num_clusters) + 'clusters.csv')

        with open(string_tokens_path, 'r') as f:
            reader = csv.reader(f)
            encoded_string_tokens = list(reader)

        return encoded_string_tokens

    def get_image_fetures(self):  ### get 128 dim embs vectors of images ####

        train_embs = np.load(self.train_embs_directory)
        train_embs = train_embs.tolist()

        return train_embs

    def get_image_metadata(self):

        file = open(self.image_metadata_directory, 'r')
        reader = csv.reader(file)
        train_labels = []
        image_names = []
        for line in reader:
            image_names.append(line[0])
            train_labels.append(line[1])

        train_labels = train_labels[1:]
        image_names = image_names[1:]

        return train_labels, image_names

    def generate_json_string_tokens_list(self, encoded_string_tokens_list, train_labels, image_names, train_embs):  #### generate json for indexing to elastic
                                                         #### search
        json_string_tokens_list = []
        for i in range(len(encoded_string_tokens_list)):
            # id = i + 1
            json_string_token = {
                'id': train_labels[i],
                'image_name': image_names[i],
                'image_url': 'empty',
                'image_encoded_tokens': encoded_string_tokens_list[i],
                'image_actual_vector': train_embs[i]
            }

            json_string_tokens_list.append(json_string_token)
            # print(json_string_tokens_list[0:10])

        return json_string_tokens_list

    def save_json_string_tokens(self, json_string_tokens_list):

        combination_name = self.encode_string_tokens_directory.split('/')[-1]
        # print(combination_name)
        if not os.path.exists(self.encode_string_tokens_directory + '/' + combination_name + '.json'):
            with open(self.encode_string_tokens_directory + '/' + combination_name + '.json', 'w') as f:
                json.dump(json_string_tokens_list, f)
                f.close()


def json_generate_main():

    json_string_tokens_generator = JsonStringTokenGenerator('encode_results_vgg', 'train_embs_VGGFace.npy',
                                                            'vn_celeb_face_recognition/train.csv', 128, 32)
    encoded_string_tokens_list = json_string_tokens_generator.get_string_tokens_list()
    train_embs = json_string_tokens_generator.get_image_fetures()
    train_labels, image_names = json_string_tokens_generator.get_image_metadata()
    json_string_tokens_list = json_string_tokens_generator.generate_json_string_tokens_list(encoded_string_tokens_list,
                                                                                            train_labels,
                                                                                            image_names,
                                                                                            train_embs)
    # print(json_string_tokens_list)

    json_string_tokens_generator.save_json_string_tokens(json_string_tokens_list)

    print('saving completed....')
    print('******************************')
        # break


if __name__ == '__main__':
    json_generate_main()





