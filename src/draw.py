import matplotlib.pyplot as plt
from src.read_data import ReadData as rd
from src.fisvdd import fisvdd
import numpy as np

path = r'..\data\jainDecision.csv'
p_data, n_data, p_label, n_label, all_data, all_label, n_index = rd(path, 3).read_data()
s_k1 = 1 / (p_data.shape[1]*np.std(p_data))
s_k2 = 1 / (n_data.shape[1]*np.std(n_data))
fd = fisvdd(p_data, 0.055)
# fd = fisvdd(p_data, s_k1)
fd.find_sv()


fd2 = fisvdd(n_data, 0.055)
# fd2 = fisvdd(n_data, s_k2)
fd2.find_sv()

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(p_data[:, 0], p_data[:, 1], s=60, alpha=1, marker='o', c='', edgecolors='steelblue') #lightseagreen , edgecolors='steelblue'
ax1.scatter(n_data[:, 0], n_data[:, 1], s=60, c='', alpha=1, marker='^', edgecolors='steelblue') #mediumseagreen, edgecolors='mediumseagreen'
ax1.set_title('Original Data')
ax1.axes.set_xlabel("(a)")

ax2 = fig.add_subplot(1, 3, 3)
# ax2.scatter(fd.sv[:, 0], fd.sv[:, 1], s=19, c='', edgecolors='steelblue', alpha=1)
ax2.scatter(fd.sv[:, 0], fd.sv[:, 1], marker='*', s=70, color='red')
# ax2.scatter(n_data[:, 0], n_data[:, 1], s=18, color='g')
# ax2.scatter(fd2.sv[:, 0], fd2.sv[:, 1], s=19, c='', edgecolors='mediumseagreen', alpha=1)
ax2.scatter(fd2.sv[:, 0], fd2.sv[:, 1], marker='+', s=70, color='red') #coral deeppink
ax2.set_title('Support Vectors')
ax2.axes.set_xlabel("(c)")

ax3 = fig.add_subplot(1, 3, 2)
ax3.scatter(p_data[:, 0], p_data[:, 1], s=60, c='', edgecolors='steelblue', alpha=1)
ax3.scatter(n_data[:, 0], n_data[:, 1], s=60, c='', marker='^', edgecolors='steelblue', alpha=1)
ax3.scatter(fd.sv[:, 0], fd.sv[:, 1], marker='*', s=50, color='red') #darkorange
ax3.scatter(fd2.sv[:, 0], fd2.sv[:, 1], marker='+', s=50, color='red', alpha=1)
ax3.set_title('Original Data with Support Vectors')
ax3.axes.set_xlabel("(b)")

fig
plt.show()


#################
#  plot result  #
#################
#

# plt.scatter(p_data[:, 0], p_data[:, 1], s=18)
# plt.title('Original Data')
# plt.savefig('original_data', dpi=300)
# plt.show()
# plt.scatter(fd.sv[:, 0], fd.sv[:, 1], s=18)
# plt.scatter(fd.sv[:, 0], fd.sv[:, 1], marker='*', s=18)
# plt.title('Support Vectors')
# plt.savefig('support_vectors', dpi=300)
# plt.show()
# plt.scatter(p_data[:, 0], p_data[:, 1], s=18)
# plt.scatter(fd.sv[:, 0], fd.sv[:, 1], marker='*', s=18)
# plt.title('Original Data with Support Vectors')
# plt.savefig('final_result', dpi=300)
# plt.show()
# plt.plot(fd.obj_val)
# plt.title('Objective Function Value')
# plt.savefig('obv', dpi=300)
# plt.show()
