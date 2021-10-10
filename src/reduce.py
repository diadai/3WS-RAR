from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from src.fisvdd import fisvdd
import src.knsi0516 as knsi0516
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import pandas
import time


class Select:

    def __init__(self, data, label, s1, class_k, label_list, para_k):
        self.data = data
        self.label = label
        self.s = s1
        # self.s2 = s2
        self.class_k = class_k
        self.label_list = label_list
        self.para_k = para_k

    def find_all_data_sv(self):
        """
        Find support vector machines for all classes separately
        :return:
        """


        for i in range(self.class_k):
            class_k_index = np.where(self.label == self.label_list[i])[0][:]
            data_k = self.data[class_k_index, :]
            # s_k = self.s #+ multiple_s*0.005
            a = np.std(data_k)
            s_k = (1 / (data_k.shape[1]*np.std(data_k))) #- 0.21 - (i*0.12)
            # s_k = 1 / data_k.shape[1]
            # s_k = self.s
            print("s:", s_k)
            print("std:", np.std(data_k))
            # if i == 0:
            #     s_k = 0.24
            # else:
            #     s_k = 0.5
            # # s_k = 1 / data_k.shape[1]
            # # print("s:", s_k)
            fd_k = fisvdd(data_k, s_k)
            fd_k.find_sv()
            fd_k._print_res()
            num_label_of_k = np.ones(fd_k.sv.shape[0]) * self.label_list[i]
            data_k_with_label = np.hstack((fd_k.sv, np.reshape(num_label_of_k, (len(num_label_of_k), 1))))
            if i == 0:
                other_data = data_k_with_label
                splicing_data = data_k_with_label
            else:
                splicing_data = np.vstack((data_k_with_label, other_data))
                other_data = splicing_data

        return splicing_data


    def get_attribute_importance_theta(self, para_radius, splicing_data):
        #splicing_data = self.find_each_class_sv()
        #splicing_data = self.random_sampling()

        # find_sv_start_time = time.time()
        # splicing_data = self.find_all_data_sv()
        # find_sv_time = time.time() - find_sv_start_time

        only_data = splicing_data[:, :-1]
        only_label = splicing_data[:, -1]

        self.temp_label_dict = Counter(only_label)
        self.temp_label_list = list(self.temp_label_dict)

        attribute_left = list(np.arange(0, only_data.shape[1], 1))
        attribute_select = []
        start = 1
        # Record last round self information
        min_information_values = 1
        first_information_values = 0
        current_attribute_relation = np.ones((only_data.shape[0], only_data.shape[0]))
        relation_list = knsi0516.GetAttributeImportance(only_data, only_label, para_radius).relation_matrix()
        while start:
            k = 0
            each_attribute_information = []

            while k < len(attribute_left):
                self_information = 0
                temp_lower_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)
                temp_upper_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)

                if len(attribute_select) == 0:
                    array_relation = relation_list[k]
                else:
                    array_relation = np.minimum(current_attribute_relation, relation_list[k])

                for i in range(only_data.shape[0]):
                    neighbor_index_i = np.where(array_relation[i, :] == 1)[0][:]
                    #neighbor_index_i = array_relation[k][i, 0: self.para_k + 1]
                    neighbor_label_set = set(only_label[neighbor_index_i])
                    if len(neighbor_label_set) == 1:
                        temp_lower_neighbor_dict[list(neighbor_label_set)[0]] += 1
                        temp_upper_neighbor_dict[list(neighbor_label_set)[0]] += 1
                    else:
                        for t in range(len(neighbor_label_set)):
                            temp_upper_neighbor_dict[list(neighbor_label_set)[t]] += 1

                for i in range(len(self.temp_label_list)):
                    temp_num_class_i = self.temp_label_dict[self.temp_label_list[i]]
                    temp_upper_approx = temp_upper_neighbor_dict[self.temp_label_list[i]]
                    temp_lower_approx = temp_lower_neighbor_dict[self.temp_label_list[i]]

                    if temp_lower_approx == 0:
                        temp_lower_approx = 1 / temp_num_class_i

                    precision = temp_lower_approx / temp_upper_approx
                    self_information = self_information + (-(1 - precision) * np.log(precision))
                each_attribute_information.append(self_information)
                k = k + 1
            # print(each_attribute_information)
            if k == 0:
                start = 0
            else:
                attribute_information_values = np.argsort(each_attribute_information, kind='stable')
                position = attribute_information_values[0]
                min_position_information_values = each_attribute_information[position]
                if len(attribute_select) == 0:
                    attribute_select.append(attribute_left[position])
                    min_information_values = 1
                    first_information_values = each_attribute_information[position]
                    current_attribute_relation = np.minimum(current_attribute_relation, relation_list[position])
                    attribute_left.pop(position)
                    relation_list.pop(position)
                else:
                    similarity = min_information_values - min_position_information_values/first_information_values
                    #print(similarity)
                    #similarity = first_information_values - min_position_information_values
                    # similarity = 1 - min_position_information_values / first_information_values
                    if similarity > 0.001:
                        attribute_select.append(attribute_left[position])
                        min_information_values = min_position_information_values / first_information_values
                        #first_information_values = min_position_information_values
                        current_attribute_relation = np.minimum(current_attribute_relation, relation_list[position])
                        attribute_left.pop(position)
                        relation_list.pop(position)
                    else:
                        start = 0
            # print(attribute_select)
        return attribute_select
