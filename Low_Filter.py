import os
import scipy.io
from scipy.signal import butter, filtfilt

# 指定原始文件夹路径和新文件夹路径
original_folder_path = r"D:\毕设数据\数据\Train"
new_folder_path =r"D:\毕设数据\MyLowData3"

# 低通滤波参数
fs = 500  # 采样频率
fc = 40   # 截止频率
order = 2  # 滤波阶数

# 低通滤波函数
def butter_lowpass_filter(data, fs, fc, order):
    Wn = fc / (fs / 2)
    b, a = butter(order, Wn, 'low')
    filtered_data = filtfilt(b, a, data)
    return filtered_data
# 获取原始文件列表
file_list = sorted([f for f in os.listdir(original_folder_path) if f.endswith('.mat')], key=lambda x: int(os.path.splitext(x)[0]))

# 循环处理前60个文件
num_files_to_process = min(60, len(file_list))

for i in range(num_files_to_process):
    # 生成原始文件的完整路径
    original_file_path = os.path.join(original_folder_path, file_list[i])

    # 读取.mat文件
    mat_data = scipy.io.loadmat(original_file_path)

    # 获取心电数据
    ecg_data = mat_data['ecg']

    # 循环处理每一行数据
    for j in range(ecg_data.shape[0]):
        # 对每一行进行低通滤波
        filtered_data = butter_lowpass_filter(ecg_data[j, :], fs, fc, order)

        # 将滤波后的数据存回原始数据
        ecg_data[j, :] = filtered_data

    # 构建新文件的完整路径
    new_file_name = file_list[i].split('.')[0] + 'L.mat'
    new_file_path = os.path.join(new_folder_path, new_file_name)

    # 将处理后的数据保存到新文件夹下
    scipy.io.savemat(new_file_path, {'ecg': ecg_data})


