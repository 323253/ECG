import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import re
from scipy.io import savemat
from keras.utils import to_categorical
#本文件准备数据集和标签
#获取文件列表
def get_file_list(folder_path, file_suffix):
    file_list = [file for file in os.listdir(folder_path) if file.endswith(file_suffix)]
    def extract_number(file_name):
        # 使用正则表达式提取结尾的数字
        match = re.search(r'(\d+)L-1\.mat$', file_name)
        return int(match.group(1))

    # 按照去除结尾后的数字进行排序
    sorted_file_list = sorted(file_list, key=extract_number)
    return sorted_file_list

#从 Excel 表格中获取标签和样本数量
def get_labels_and_sample_counts(excel_path, file_names):
    df = pd.read_excel(excel_path)
    labels = []
    sample_counts = []

    for file_name in file_names:
        #分上凸下凹两个类型
        ste_value = df[df['name'] == re.findall(r'\d+',file_name)[0]+'.mat']['STE'].values[0]
        std_value = df[df['name'] == re.findall(r'\d+',file_name)[0]+'.mat']['STD'].values[0]
        others_value = df[df['name'] == re.findall(r'\d+',file_name)[0]+'.mat']['Others'].values[0]
        if ste_value == 1 and std_value==1:
            labels.append(3)
        elif ste_value == 1:
            labels.append(1)
        elif others_value == 1:
            labels.append(0)
        else:
            labels.append(2)

        # 获取样本数量
        file_path = os.path.join(folder_path, file_name)
        mat_data = loadmat(file_path)
        sample_count = mat_data['variable_name'].shape[0]
        sample_counts.append(sample_count)

    return np.array(labels), np.array(sample_counts)

# Step 3: 从 .mat 文件中加载数据
def load_data_from_mat(file_path):
    mat_data = loadmat(file_path)
    mats_data=mat_data['variable_name']
    celld=[]
    for cell in mats_data:
        # cells=[item for item in cell]
        celld.append(cell)
    return  celld # 请替换成实际的数据键名

# Step 4: 构建 CNN 模型

# Step 5: 获取文件列表
folder_path = r"D:\毕设数据\ST_Select_Data_New"
file_suffix = '-1.mat'
file_names = get_file_list(folder_path, file_suffix)

# Step 6: 从 Excel 表格中获取标签和样本数量
excel_path = r"D:\毕设数据\数据\Train.xlsx"
labels, sample_counts = get_labels_and_sample_counts(excel_path, file_names)

# Step 7: 加载数据并生成标签
input_data = []
label_data = []

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    mat_datas = load_data_from_mat(file_path)
    input_data.extend(mat_datas)
    # # 将每个样本和对应标签复制 sample_count 次
    # input_data.extend([mat_data] * 1)
    # label_data.extend([labels] * sample_count)
labelss = [label for label, count in zip(labels, sample_counts) for _ in range(count)]


# input_data = np.array(input_data)
label_data = np.array(labelss)
label_data_hot=to_categorical(label_data,num_classes=4)
# print('输入ST段数量：',len(input_data),'标签数量：',len(label_data))
# file_path1=r"D:\毕设数据\input_lables\input_data.mat"
# file_path2=r"D:\毕设数据\input_lables\label_data.mat"
file_path3=r"D:\毕设数据\input_lables\label_data_hot.mat"

# savemat(file_path1,{'input_data':input_data})
# savemat(file_path2,{'label_data':label_data})
savemat(file_path3,{'label_data_hot':label_data_hot})