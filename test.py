from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
import pandas

class Domain:
    def __init__(self, _name, _label, _length, _num, _entropy):
        self.name = _name
        self.label = _label
        self.length = _length
        self.num = _num
        self.entropy = _entropy

    def return_data(self):
        return [self.length, self.num, self.entropy]

    def return_label(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def Number(str):
    num = 0
    for i in str:
        if i.isdigit():
            num = num + 1
    return num

def Entropy(str):
    e = 0.0
    sum = 0
    letter = [0] * 26
    str = str.lower()
    for i in range(len(str)):
        if str[i].isalpha():
            letter[ord(str[i]) - 97] += 1
            sum = sum + 1
    for i in range(26):
        temp = float(letter[i]) / sum
        if temp > 0:
            e += -(temp * math.log(temp, 2))
    return e

def init_data(filename, list):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            segmentation = line.split(",")
            name = segmentation[0]
            if len(segmentation) > 1:
                label = segmentation[1]
            else:
                label = "?"
            length = len(name)
            num = Number(name)
            entropy = Entropy(name)
            list.append(Domain(name, label, length, num, entropy))

if __name__ == '__main__':
    list1 = []
    init_data("train.txt", list1)
    feature_matrix = []
    label_list = []
    for item in list1:
        feature_matrix.append(item.return_data())
        label_list.append(item.return_label())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(feature_matrix, label_list)

    list2 = []
    init_data("test.txt", list2)
    with open("result.txt", "w+") as f:
        for i in list2:
            f.write(i.name)
            f.write(",")
            if clf.predict([i.return_data()])[0] == 0:
                f.write("notdga\n")
            else:
                f.write("dga\n")

