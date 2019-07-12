import csv
import numpy as np

path = './vn_celeb_face_recognition/train.csv'

file = open(path, 'r')
reader = csv.reader(file)
train_data = []
for line in reader:
    train_data.append(line[1])

train_data = train_data[1:]
# print(len(train_data))

indices = [i for i, x in enumerate(train_data) if x == "399"]
print(indices)