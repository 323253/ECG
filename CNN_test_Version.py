import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

#获取文件列表
def get_file_list(folder_path, file_suffix):
    file_list = [file for file in os.listdir(folder_path) if file.endswith(file_suffix)]
    return file_list

#从 Excel 表格中获取标签和样本数量
def get_labels_and_sample_counts(excel_path, file_names):
    df = pd.read_excel(excel_path)
    labels = []
    sample_counts = []

    for file_name in file_names:
        #分上凸下凹两个类型
        ste_value = df[df['name'] == file_name[0]+'.mat']['STE'].values[0]
        others_value = df[df['name'] == file_name[0]+'.mat']['Others'].values[0]

        if ste_value == 1:
            labels.append(1)
        elif others_value == 1:
            labels.append(0)
        else:
            labels.append(2)

        # 获取样本数量
        file_path = os.path.join(folder_path, file_name)
        mat_data = loadmat(file_path)
        sample_count = mat_data['variable_name'].shape[0]  # 请替换成实际的数据键名
        sample_counts.append(sample_count)

    return np.array(labels), np.array(sample_counts)

# Step 3: 从 .mat 文件中加载数据
def load_data_from_mat(file_path):
    mat_data = loadmat(file_path)
    mats_data=mat_data['variable_name']
    celld=[]
    for cell in mats_data:
        cells=[item for item in cell]
        celld.append(cells)
    return  celld # 请替换成实际的数据键名

# Step 4: 构建 CNN 模型
def CNN_Model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(48, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # 3个类别，可以根据实际情况调整

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 5: 获取文件列表
folder_path = r"D:\毕设数据\ST_Select_Data"
file_suffix = '-1.mat'
file_names = get_file_list(folder_path, file_suffix)

# Step 6: 从 Excel 表格中获取标签和样本数量
excel_path = r"D:\毕设数据\数据\Train.xlsx"
labels, sample_counts = get_labels_and_sample_counts(excel_path, file_names)

# Step 7: 加载数据并生成标签
input_data = []
label_data = []

for file_name, sample_count in zip(file_names, sample_counts):
    file_path = os.path.join(folder_path, file_name)
    mat_data = load_data_from_mat(file_path)

    # 将每个样本和对应标签复制 sample_count 次
    input_data.extend([mat_data] * 1)
    label_data.extend([labels] * sample_count)

input_data = np.array(input_data)
label_data = np.array(label_data)
print('input_data,label_data,finish')
# Step 8: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)
print("split")
# Step 9: 构建 CNN 模型
input_shape = input_data.shape[1:]  # 根据实际情况调整
model = CNN_Model(input_shape)
print("CNN_Model finish")
# Step 10: 训练模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
print("train")