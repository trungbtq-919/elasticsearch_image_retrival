import os
import json
import numpy as np
import csv

class JsonStringTokenGenerator(object):

    def __init__(self, encoded_string_tokens_list, train_embs, train_labels, image_names):
        
        self.encoded_string_tokens_list = encoded_string_tokens_list
        self.train_embs = train_embs
        self.train_labels = train_labels
        self.image_names = image_names


    def generate_json_string_tokens_list(self):

        json_string_tokens_list = []
        for i in range(len(self.encoded_string_tokens_list)):
            # id = i + 1
            json_string_token = {
                'id': self.train_labels[i],
                'image_name': self.image_names[i],
                'image_url': 'empty',
                'image_actual_vector': self.train_embs[i],
                'image_encoded_tokens': self.encoded_string_tokens_list[i]
            }

            json_string_tokens_list.append(json_string_token)
            # print(json_string_tokens_list[0:10])

        return json_string_tokens_list


def get_list_directory(path): ## get list of directory containing encoded string tokens

    directory_list = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))
    return directory_list

# directory = directory_list[0]
# print(directory)


def get_string_tokens_list(directory):
    """ Get encoded string tokens from csv files.
    """

    string_tokens_paths = list()

    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            string_tokens_paths.append(os.path.join(root, name))

    for string_tokens_path in string_tokens_paths:
        if '.csv' in string_tokens_path:
            with open(string_tokens_path, 'r') as f:
                reader = csv.reader(f)
                encoded_string_tokens = list(reader)

    return encoded_string_tokens


def get_image_fetures(features_csv_path): ### get 128 dim embs vectors of images ####

    train_embs = np.load(features_csv_path)
    train_embs = train_embs.tolist()

    return train_embs


def get_metadata(image_path):

    file = open(image_path, 'r')
    reader = csv.reader(file)
    train_labels = []
    image_names = []
    for line in reader:
        image_names.append(line[0])
        train_labels.append(line[1])

    # First line in .csv file is "image + label"
    # So *start_index* should be started from 1 instead of 0.
    image_names = image_names[1:]
    train_labels = train_labels[1:]

    return image_names, train_labels


def save_json_string_tokens(directory, json_string_tokens_list):

    combination_name = directory.split('/')[-1]
    # print(combination_name)
    if not os.path.exists(directory+'/'+combination_name+'.json'):
        with open(directory+'/'+combination_name+'.json', 'w') as f:
            json.dump(json_string_tokens_list, f)
            f.close()


def main():
    
    path = './encode_results'
    features_csv_path = './train_embs.npy'
    
    directory_list = get_list_directory(path)
    for directory in directory_list:

        print('start to generate and save string tokens with {} as json'.format(directory.split('/')[-1]))

        encoded_string_tokens = get_string_tokens_list(directory)

        # print(len(encoded_string_tokens))

        train_embs = get_image_fetures(features_csv_path)
        # print(len(train_embs))

        image_id_path = './vn_celeb_face_recognition/train.csv'
        train_labels = get_image_id(image_id_path)

        json_string_tokens_generator = JsonStringTokenGenerator(encoded_string_tokens, train_embs, train_labels)
        json_string_tokens_list = json_string_tokens_generator.generate_json_string_tokens_list()
        # print(json_string_tokens_list)

        save_json_string_tokens(directory, json_string_tokens_list)

        print('saving completed....')
        print('******************************')
        # break


if __name__ == '__main__':
    main()