import os
import json
import numpy as np
import csv

path = './encode_results'
features_csv_path = './train_embs.npy'


def get_list_directory(path):
    """Get list of directory containing encoded string tokens
    """

    directory_list = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))
    return directory_list

# directory = directory_list[0]
# print(directory)


def get_string_tokens_list(directory): ## get encoded string tokens from csv files ##
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

def generate_json_body_index(encoded_string_tokens, train_embs, train_labels, image_names):
    """ Generate json for indexing to elasticsearch
    """
                        
    json_body_index_list = []
    for i in range(len(encoded_string_tokens)):
        # id = i + 1
        json_body_index = {
            "label": train_labels[i],
            "image_name": image_names[i],
            "image_url": "empty",
            "embedding_vector": train_embs[i],
            "string_token": encoded_string_tokens[i],
            # 'image_encoded_tokens': encoded_string_tokens[i],
            # 'image_actual_vector': train_embs[i]
        }

        json_body_index_list.append(json_body_index)
        # print(json_string_tokens_list[0:10])

    return json_body_index_list


def save_json_string_tokens(directory, json_string_tokens_list):

    combination_name = directory.split('/')[-1]
    # print(combination_name)
    # if not os.path.exists(directory+'/'+combination_name+'.json'):
    with open(directory + '/' + combination_name + '.json', 'w') as f:
        json.dump(json_string_tokens_list, f)
        # f.close() <-- meaningless


def generate_json_main():
    directory_list = get_list_directory(path)
    for directory in directory_list:

        print('start to generate and save string tokens with {} as json'.format(directory.split('/')[-1]))

        encoded_string_tokens = get_string_tokens_list(directory)

        # print(len(encoded_string_tokens))

        train_embs = get_image_fetures(features_csv_path)
        # print(len(train_embs))

        image_path = './vn_celeb_face_recognition/train.csv'
        image_names, train_labels = get_metadata(image_path)

        json_string_tokens_list = generate_json_body_index(encoded_string_tokens, train_embs, train_labels, image_names)

        save_json_string_tokens(directory, json_string_tokens_list)

        print('saving completed....')
        print('******************************')
        # break

def main():
    """
    image_id_path = './vn_celeb_face_recognition/train.csv'
    train_labels = get_image_id(image_id_path)
    print(train_labels)
    """
    generate_json_main()

if __name__ == '__main__':
    main()





