import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import os
import pywt
from scipy.signal import find_peaks

def l_filter_ecg(ecg_signal,cutoff_frequency, sampling_rate):
    # 设计一个低通Butterworth滤波器
    b, a = signal.butter(4, cutoff_frequency / (0.5 * sampling_rate), btype='low')

    # 应用滤波器
    filtered_ecg = signal.filtfilt(b, a, ecg_signal)
    plt.plot(filtered_ecg)
    plt.title("ECG-L-Data")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()
    return filtered_ecg
def m_filter_ecg(signal_data, sampling_rate,L_rate,H_rate):
    # 设计一个带通滤波器，通常ST段的频率范围是0.05 Hz到0.5 Hz fs采样率
    b, a = signal.butter(2, [L_rate, H_rate], btype='band', fs=sampling_rate)

    # 应用滤波器
    filtered_signal = signal.filtfilt(b, a, signal_data)
    plt.plot(filtered_signal)
    plt.title("ECG-M-Data")
    plt.show()
    return filtered_signal

# 小波变换矫正基线漂移
def remove_baseline_drift(ecg_data):
    # 使用小波变换进行分解和重构
    coeffs = pywt.wavedec(ecg_data, 'db1', level=3)  # 5层小波分解
    # coeffs[1:] = (0.0, ) * len(coeffs[1:])  # 将高频成分设为0
    # 将高频小波系数置零
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])

    baseline_corrected_ecg = pywt.waverec(coeffs, 'db1')  # 重构信号
    plt.plot(baseline_corrected_ecg)
    plt.title("ECG-B-Data")
    plt.show()
    return baseline_corrected_ecg

def QRS_detect(ecg_data,sampling_rate):
    t = np.arange(0, 7500, 1 / sampling_rate)
    # 计算阈值（例如，取信号的90th百分位数）
    threshold_percentile = 90
    threshold = np.percentile(ecg_data, threshold_percentile)

    # 寻找超过阈值的点
    o_qrs_indices = np.where(ecg_data > threshold)[0]
    # 筛选掉左右两侧斜率方向相同的点
    qrs_indices_1 = []
    qrs_indices=[]

    for qrs_index in o_qrs_indices:
        if qrs_index > 0 and qrs_index < len(ecg_data) - 1:
            # 计算左侧和右侧的斜率
            slope_left = ecg_data[qrs_index] - ecg_data[qrs_index - 1]
            slope_right = ecg_data[qrs_index + 1] - ecg_data[qrs_index]

            # 判断斜率方向是否相反
            if slope_left==0:
                if np.sign(slope_left) != np.sign(slope_right):
                    qrs_indices_1.append(qrs_index)
    for qrs_indice in qrs_indices_1:
        if qrs_indice > 0 and qrs_indice < len(ecg_data) - 1:
            slope_left = ecg_data[qrs_indice] - ecg_data[qrs_indice - 12]
            slope_right = ecg_data[qrs_indice + 1] - ecg_data[qrs_indice]
            if np.sign(slope_left) != np.sign(slope_right):
                qrs_indices.append(qrs_indice)



    # 绘制原始心电图和QRS检测结果
    plt.figure(figsize=(12, 6))
    plt.plot(t[:len(ecg_data)], ecg_data, label='Original ECG')
    plt.scatter(t[qrs_indices], ecg_data[qrs_indices], c='red', label='Detected QRS Peaks')
    plt.axhline(y=threshold, color='gray', linestyle='--', label=f'{threshold_percentile}th Percentile Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    # 提取一个周期的心电信号
    # if len(qrs_indices) >= 2:
    #     # # 计算心拍周期
    #     # heartbeat_period = np.diff(qrs_indices).mean()
    #
    #     # 选择一个心拍周期的时间窗口
    #     start_index = qrs_indices[0]
    #     end_index=qrs_indices[1]
    #     # end_index = start_index + int(heartbeat_period)
    #
    #     # 绘制原始心电图和提取的一个周期的信号
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(t[:len(ecg_data)], ecg_data, label='Original ECG')
    #     t=t[:len(ecg_data)]
    #     plt.plot(t[start_index:end_index], ecg_data[start_index:end_index], label='One Heartbeat')
    #     plt.scatter(t[qrs_indices], ecg_data[qrs_indices], c='red', label='Detected QRS Peaks')
    #     plt.axhline(y=threshold, color='gray', linestyle='--', label=f'{threshold_percentile}th Percentile Threshold')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Amplitude')
    #     plt.legend()
    #     plt.show()
    # else:
    #     print("QRS complexes not found.")

    # 提取心跳周期
    #提取心跳周期
    heart_onces=[]
    for i in range(len(qrs_indices) - 1):
        start_index = qrs_indices[i]
        end_index = qrs_indices[i + 1]

        # 提取心跳周期
        heartbeat = ecg_data[start_index:end_index]

        # 将心跳周期添加到列表中
        heart_onces.append(heartbeat)
    plt.figure(figsize=(12, 6))
    plt.plot(heart_onces[0])
    plt.show()




def detect_st_segment(ecg_signal, sampling_rate):
    # 使用差分法计算斜率
    derivative = np.diff(ecg_signal)

    # 设置一个阈值来检测ST段的斜率
    threshold = 0.1

    # 找到超过阈值的索引
    st_segment_indices = np.where(derivative > threshold)[0]

    return st_segment_indices

def getData(folder_path):

    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    select_files=[file for file in files if file.endswith(".mat") and file[:-4].isdigit()]
    if len(select_files)>30:
        select_files=select_files[:30]
        for file_name in select_files:
            file_path=os.path.join(folder_path,file_name)#拼成完整路径
            # 读取MATLAB .mat文件
            mat_data = scipy.io.loadmat(file_path)
            # 查看.mat文件中的变量信息
            mat_variables = scipy.io.whosmat(file_path)
            print(mat_variables)
            # .mat文件中有一个名为 "ecg" 的变量，获取其中的第一行数据
            first_row_data = mat_data['ecg'][0, :]
            print(1)##
        print(first_row_data)
        plt.plot(first_row_data)
        plt.title("ECG Data")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()
        return first_row_data



# 指定文件夹路径
folder_path = r"D:\毕设数据\数据\Test"
#getData(folder_path)

#应用低通
l_filter_ecg=l_filter_ecg(getData(folder_path),50,1000)
print(2)
print(l_filter_ecg)
remove_baseline_drift=remove_baseline_drift(l_filter_ecg)
print(3)
print(remove_baseline_drift)
#m_filter_ecg=m_filter_ecg(remove_baseline_drift,2000,0.05,0.5)
QRS_detect(remove_baseline_drift,1000)
# plt.plot(first_row_data)
# plt.title("ECG Data")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")
# plt.show()
# # 生成示例心电图数据
# sampling_rate = 1000  # 采样率
# t = np.arange(0, 10, 1 / sampling_rate)
# ecg_data = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.sin(2 * np.pi * 1.5 * t)
#
# # 应用滤波器
# filtered_ecg = filter_ecg(ecg_data, sampling_rate)
#
# # 检测ST段
# st_segment_indices = detect_st_segment(filtered_ecg, sampling_rate)
#
# # 绘制心电图和标记ST段
# plt.figure(figsize=(12, 6))
# plt.plot(t, ecg_data, label='Original ECG')
# plt.plot(t, filtered_ecg, label='Filtered ECG')
# plt.scatter(t[st_segment_indices], filtered_ecg[st_segment_indices], color='red', label='ST Segment')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()
