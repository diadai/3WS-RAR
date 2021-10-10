import numpy as np

from src.read_data import ReadData as rd
import src.cross_validation as cr
from src.compare import Comparison
from src.reduce import Select
#from some_test.svm_reduction import Select
import time
from collections import Counter

path = r'..\data\Dry_Bean_Dataset.csv'

all_data, all_label = rd(path, 1).read_data()
class_dict = Counter(all_label[:, 0])
class_list = []
for item in class_dict:
    class_list.append(item)

k_folds = 10

temp_train_index, temp_test_index = cr.cross_validation(all_data.shape[0], k_folds)
score = []
s = 0.6
class_k = len(class_list)

temp_accuracy_comparison = []
temp_f1_score = []
temp_cost_time = []

Select = Select(all_data, all_label, s, class_k, class_list, para_k=3)
para_radius = 0.025
while para_radius < 0.4:
    print(para_radius)
    # # BIDC
    temp_accuracy_selected_comparison = []
    temp_f1_selected_score = []
    temp_selected_cost_time = []

    print(para_radius)
    if para_radius == 0.025:
        start_time1 = time.time()
        # Select = Select(all_data, all_label, s, class_k, class_list, para_k=3)
        find_sv_data = Select.find_all_data_sv()
        time1 = time.time() - start_time1

    start_time2 = time.time()
    temp_select_attribute_index = Select.get_attribute_importance_theta(para_radius, find_sv_data)
    print("attribute_index", temp_select_attribute_index)
    time2 = time.time() - start_time2
    end_time = time2 + time1
    print("time", end_time)
    for j in range(10):
        #print(j)
        train_data = all_data[temp_train_index[j]]
        train_label = all_label[temp_train_index[j]]

        test_data = all_data[temp_test_index[j]]
        test_label = all_label[temp_test_index[j]]
        ##Initial data experiment results
        if para_radius == 0.025:
            temp_accuracy_list_original, temp_f1_score_list_original, temp_cost_time_original = Comparison(train_data,
                                                             train_label, test_data, test_label).comparison()
            temp_accuracy_comparison.append(temp_accuracy_list_original)
            temp_f1_score.append(temp_f1_score_list_original)
            temp_cost_time.append(temp_cost_time_original)

        temp_accuracy_list_select, temp_f1_score_list_select, temp_cost_time_select = Comparison(train_data[:, temp_select_attribute_index],
                                    train_label, test_data[:, temp_select_attribute_index], test_label).comparison()

        temp_accuracy_selected_comparison.append(temp_accuracy_list_select)
        temp_f1_selected_score.append(temp_f1_score_list_select)
        temp_selected_cost_time.append(temp_cost_time_select)

    for k in range(2):
        final_score_comparison = []
        final_f1_score = []
        final_cost_time = []

        final_score_selected_comparison = []
        final_f1_score_selected = []
        final_cost_time_selected = []

        for z in range(10):
            final_score_comparison.append(temp_accuracy_comparison[z][k])
            final_f1_score.append(temp_f1_score[z][k])
            final_cost_time.append(temp_cost_time[z][k])

            final_score_selected_comparison.append(temp_accuracy_selected_comparison[z][k])
            final_f1_score_selected.append(temp_f1_selected_score[z][k])
            final_cost_time_selected.append(temp_selected_cost_time[z][k])

        score_test_comparison = sum(final_score_comparison) / 10
        temp_standard_comparison = np.std(final_score_comparison)

        f1 = sum(final_f1_score)/10
        temp_f1_standard = np.std(final_f1_score)
        average_time = sum(final_cost_time)/10
        print("o acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (score_test_comparison,
                                                    temp_standard_comparison, f1, temp_f1_standard, average_time))

        score_test_selected_comparison = sum(final_score_selected_comparison) / 10
        temp_standard_selected_comparison = np.std(final_score_selected_comparison)
        f1_selected = sum(final_f1_score_selected) / 10
        temp_f1_sel_standard = np.std(final_f1_score_selected)
        average_time_selected = sum(final_cost_time_selected) / 10
        print("s acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (score_test_selected_comparison,
                            temp_standard_selected_comparison, f1_selected, temp_f1_sel_standard, average_time_selected))
    para_radius += 0.025









