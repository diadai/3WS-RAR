import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class GetAttributeImportance:
    """
    class of get attribute importance
    """

    def __init__(self, train_data, train_label, para_k):

        self.train_data = train_data
        self.train_label = train_label
        self.k = para_k
        self.instance_length = len(self.train_data[:, 0])
        self.temp_col_length = len(self.train_data[0])
        self.temp_label_dict = Counter(self.train_label)
        self.temp_label_list = list(self.temp_label_dict)

    #def calculate_distance(self):

    def relation_matrix(self):
        """

        :return: Neighbor relationship state matrix
        """
        #new_array = np.zeros((self.instance_length, self.instance_length))
        relation_list = []
        for i in range(self.temp_col_length):
            neighbor_status = np.zeros((self.instance_length, self.instance_length))
            distance = cdist(self.train_data[:, i: i+1], self.train_data[:, i: i+1], metric='euclidean')
            # sort_index = np.argsort(distance)
            for j in range(self.instance_length):
                true_index = np.where(distance[j][:] < self.k)[0][:]
                neighbor_status[j][true_index] = 1
            relation_list.append(neighbor_status)

        return relation_list

    def relation_matrix_2(self, para_select, para_left):
        """

        :param para_select:
        :param para_left:
        :return:
        """
        relation_list = []
        for i in range(len(para_left)):
            copy_select = para_select.copy()
            copy_select.append(para_left[i])
            neighbor_status = np.zeros((self.instance_length, self.instance_length))
            distance = cdist(self.train_data[:, copy_select], self.train_data[:, copy_select], metric='euclidean')
            diag_array = np.diag([-100] * self.train_data.shape[0])
            sort_index = np.argsort(distance+diag_array, kind='stable')
            # for j in range(self.instance_length):
            #     for k in range(self.k):
            #         neighbor_status[j][sort_index[j][k + 1]] = 1
            relation_list.append(sort_index)
        return relation_list







