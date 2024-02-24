import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import os
import pywt
from scipy.signal import find_peaks
import math

def QRS_detect_Blutty(ecg_signals, tp):
    ecg_segmentd=[]
    for ecg_signal in ecg_signals:
        threshold_percent=tp
        # 计算振幅的阈值（正值）
        amplitude_threshold = np.percentile(ecg_signal[ecg_signal > 0], threshold_percent)

        # 找到正振幅超过阈值的区域
        high_positive_amplitude_indices = np.where((ecg_signal > 0) & (ecg_signal > amplitude_threshold))[0]

        # 将连续的区域存储为列表元素
        continuous_regions = []
        current_region = [high_positive_amplitude_indices[0]]
        for i in range(1, len(high_positive_amplitude_indices)):
            if high_positive_amplitude_indices[i] - high_positive_amplitude_indices[i - 1] == 1:
                # 如果是连续的索引，继续添加到当前区域
                current_region.append(high_positive_amplitude_indices[i])
            else:
                # 如果不连续，将当前区域添加到列表，并开始新的区域
                continuous_regions.append(current_region)
                current_region = [high_positive_amplitude_indices[i]]

        # 处理最后一个区域
        continuous_regions.append(current_region)
        Median_Value=[]
        # # 绘制心电图和连续正振幅超过95%的区域（橙色线条）
        # plt.plot(ecg_signal, label='ECG Signal')
        for region in continuous_regions:
            #region_data = ecg_signal[region[0]:region[-1] + 1]
            median_value = np.median(region)
            Median_Value.append(median_value)
        #     plt.axvline(np.median(region), color='red', linestyle='--', label=f'Median: {median_value:.2f}')
        #     plt.plot(region, region_data, 'orange', linewidth=2)
        # plt.title('ECG Signal with Continuous High Positive Amplitude Regions ( > 95%)')
        # plt.legend()
        # plt.show()

        # 将心电图信号按切割时间点切割成多个片段，并放入列表中
        ecg_segments = [ecg_signal[int(Median_Value[i]):int(Median_Value[i + 1])]
                        for i in range(len(Median_Value) - 1)]
        #ecg_segments.append(ecg_signal[int(Median_Value[-1] ):])
        # plt.plot(ecg_segments[0])
        # plt.title("One-Heart-Data")
        # plt.xlabel("Sample")
        # plt.ylabel("Amplitude")
        # plt.show()
        ecg_segmentd.append(ecg_segments)
    print(1)
    # plt.plot(ecg_segmentd[0][0])
    # plt.show()
    # print(1)
    return ecg_segmentd
def Judge_cust(ecg_signaltd):
    new_ecg_signal=[]
    for ecg_signals in ecg_signaltd:
        start_index=math.ceil(len(ecg_signals)/3)
        end_index=2*math.ceil(len(ecg_signals)/3)
        cust_signal=ecg_signals[start_index:end_index]
        cust_sign=max(cust_signal)
        former_signal=ecg_signals[:start_index]
        former_sign=max(former_signal)
        behind_signal=ecg_signals[end_index:]
        behind_sign=max(behind_signal)
        if former_sign>cust_sign*1.3 and behind_sign>cust_sign*1.3:
            new_ecg_signal.append(cust_signal)
            print("前后去除")
        elif former_sign>cust_sign*1.3:
            new_ecg_signal.append(ecg_signals[start_index:])
            print("前端去除")
        elif behind_sign > cust_sign * 1.3:
            new_ecg_signal.append(ecg_signals[:end_index])
            print("后端去除")
        else:
            new_ecg_signal.append(ecg_signals)
            print("完整")


    return new_ecg_signal


def Delete_abnomal_Beat(heartbeatd):
    filtered_nomals=[]
    for heartbeats in heartbeatd:
        # 计算每个元素的长度
        lengths = [len(item) for item in heartbeats]

        # 利用标准差与正态分布关系排除异常选择心跳周期
        mean_length = sum(lengths) / len(lengths)
        std_dev = (sum((length - mean_length) ** 2 for length in lengths) / len(lengths)) ** 0.5

        # 设置一个阈值，如果元素长度与均值的差异大于阈值的倍数，就认为它长度异常
        threshold = 2.0

        # 找到长度异常的元素的索引
        outliers = [index for index, length in enumerate(lengths) if abs(length - mean_length) > threshold * std_dev]

        # 从列表中去掉长度异常的元素
        filtered_nomal = [heartbeats[index] for index in range(len(heartbeats)) if index not in outliers]
    filtered_nomals.append(filtered_nomal)
    return filtered_nomals


def getData_l(folder_path):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    select_files=[file for file in files if file.endswith("L.mat") and file[:-5].isdigit()]
    # 对文件名按照数字进行排序
    select_files.sort(key=lambda x: int(x[:-5]))
    if len(select_files)>30:
        select_files=select_files[:30]
        for file_name in select_files:
            print(file_name)
            file_path=os.path.join(folder_path,file_name)#拼成完整路径
            first_row_data = []
            # 读取MATLAB .mat文件
            mat_data = scipy.io.loadmat(file_path)

            rows=mat_data['filtered_ecg']
            for row in rows:
                first_row_data.append(row)
            print(1)
            Judgecust = Judge_cust(first_row_data)
            QRS_detect = QRS_detect_Blutty(Judgecust, 98)
            Delete_abnomal = Delete_abnomal_Beat(QRS_detect)

            # file=r"D:\毕设数据\QRSdata"
            # scipy.io.savemat(file, {'variable_name': QRS_detect[0]})
            # print('QRS_detect finish')

            #first_row_data = mat_data['filtered_ecg'][0, :]
        return first_row_data

# 指定文件夹路径
folder_path_l = r"D:\毕设数据\MyLowData"
folder_path=r"D:\毕设数据\数据\Train"
#getData(folder_path)
getData_l=getData_l(folder_path_l)