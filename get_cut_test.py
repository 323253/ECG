import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import os
import pywt
from scipy.signal import find_peaks

def get_stableData(ecg_signal, discard_front, discard_end,sampling_rate):
    front_samples = int(discard_front * sampling_rate)
    end_samples = int(discard_end * sampling_rate)

    stable_segment = ecg_signal[front_samples: -end_samples]

    return stable_segment
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
    b, a = signal.butter(1, [L_rate, H_rate], btype='band', fs=sampling_rate)

    # 应用滤波器
    filtered_signal = signal.filtfilt(b, a, signal_data)
    plt.plot(filtered_signal)
    plt.title("ECG-M-removeNoice-Data")
    plt.show()
    return filtered_signal

# 小波变换矫正基线漂移
def remove_baseline_drift(ecg_data):
    # 使用小波变换进行分解和重构
    coeffs = pywt.wavedec(ecg_data, 'db4', level=5)  # 5层小波分解
    # coeffs[1:] = (0.0, ) * len(coeffs[1:])  # 将高频成分设为0
    # 将高频小波系数置零
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])

    baseline_corrected_ecg = pywt.waverec(coeffs, 'db4')  # 重构信号
    plt.plot(baseline_corrected_ecg)
    plt.title("ECG-B-Data")
    plt.show()
    return baseline_corrected_ecg
#小波去噪
def Daubechies_remove_noise(ecg_signal, wavelet, level):
    wavelet=wavelet
    level=level
    #小波分解系数
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # 阈值化去噪
    threshold = 0.1  # 调整阈值以控制去噪程度
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]

    # 重构信号
    denoised_ecg = pywt.waverec(coeffs, wavelet)

    t = np.arange(0, 7500, 1 / 2000)
    t = t[:len(ecg_signal)]
    # 绘制原始心电图和去噪后的信号
    plt.figure(figsize=(10, 6))
    plt.plot(t, ecg_signal, label='Original ECG Signal')
    plt.plot(t, denoised_ecg, label='Denoised ECG Signal', linewidth=2)
    plt.title('Original and Denoised ECG Signal using Wavelet Transform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    return denoised_ecg

def L_filter_array(ecg_signals,cutoff_frequency, sampling_rate):
    # 设计一个低通Butterworth滤波器
    b, a = signal.butter(2, cutoff_frequency / (0.5 * sampling_rate), btype='low')
    filtered_ecg=[]
    # 应用滤波器
    for ecg_signal in ecg_signals:
        filtered_ecg.append(signal.filtfilt(b, a, ecg_signal))

    # plt.figure(figsize=(12, 6))
    # plt.plot(filtered_ecg[0])
    # plt.title(f'One-Heart-Beat-LowFilter {cutoff_frequency}HZ')
    # plt.show()
    return filtered_ecg
def M_filter_array(ecg_signals,L_rate, H_rate,sampling_rate):
    # 设计一个带通Butterworth滤波器
    b, a = signal.butter(2, [L_rate, H_rate], btype='band', fs=sampling_rate)
    filtered_ecg=[]
    ecg_signals[0]
    # 应用滤波器
    for ecg_signal in ecg_signals:
        filtered_ecg.append(signal.filtfilt(b, a, ecg_signal,padlen=len(ecg_signal)-1))

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_ecg[0])
    plt.show()
    return filtered_ecg


def QRS_detect(ecg_data,sampling_rate):
    t = np.arange(0, 7500, 1 / sampling_rate)
    # 计算阈值（例如，取信号的90th百分位数）
    threshold_percentile = 94
    threshold = np.percentile(ecg_data, threshold_percentile)

    # 寻找超过阈值的点
    o_qrs_indices = np.where(ecg_data > threshold)[0]
    # 筛选掉左右两侧斜率方向相同的点
    qrs_indices_1 = []
    qrs_indices_0 = []
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
                qrs_indices_0.append(qrs_indice)
    for qrs_indice in qrs_indices_0:
        if qrs_indice > 2 and qrs_indice < len(ecg_data) - 1:
            former=qrs_indices_0[qrs_indices_0.index(qrs_indice)-1]
            distance=qrs_indice-former
            if distance>250 or ecg_data[qrs_indice]>ecg_data[former]:
                qrs_indices.append(qrs_indice)

    # 绘制原始心电图和QRS检测结果
    # for qrs_index in o_qrs_indices:
    #     if qrs_index > 0 and qrs_index < len(ecg_data) - 1:
    #             # 计算左侧和右侧的斜率
    #             slope_left = ecg_data[qrs_index] - ecg_data[qrs_index - 6]
    #             slope_right = ecg_data[qrs_index]- ecg_data[qrs_index + 6]
    #
    #             # 判断斜率方向是否相反
    #             if slope_left<0 and slope_right<0:
    #                 qrs_indices.append(qrs_index)
#有效绘图开始
    # plt.figure(figsize=(12, 6))
    # plt.plot(t[:len(ecg_data)], ecg_data, label='Original ECG')
    # plt.scatter(t[qrs_indices], ecg_data[qrs_indices], c='red', label='Detected QRS Peaks')
    # plt.axhline(y=threshold, color='gray', linestyle='--', label=f'{threshold_percentile}th Percentile Threshold')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()
    #有效绘图结束

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
    # plt.figure(figsize=(12, 6))
    # plt.plot(heart_onces[0])
    # plt.show()

def QRS_detect_Blutty(ecg_signal, tp):
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
    # 绘制心电图和连续正振幅超过95%的区域（橙色线条）
    plt.plot(ecg_signal, label='ECG Signal')
    for region in continuous_regions:
        region_data = ecg_signal[region[0]:region[-1] + 1]
        median_value = np.median(region)
        Median_Value.append(median_value)
        plt.axvline(np.median(region), color='red', linestyle='--', label=f'Median: {median_value:.2f}')
        plt.plot(region, region_data, 'orange', linewidth=2)
    plt.title('ECG Signal with Continuous High Positive Amplitude Regions ( > 95%)')
    plt.legend()
    plt.show()

    # 将心电图信号按切割时间点切割成多个片段，并放入列表中
    ecg_segments = [ecg_signal[int(Median_Value[i]):int(Median_Value[i + 1])]
                    for i in range(len(Median_Value) - 1)]
    ecg_segments.append(ecg_signal[int(Median_Value[-1] ):])
    # plt.plot(ecg_segments[0])
    # plt.title("One-Heart-Data")
    # plt.xlabel("Sample")
    # plt.ylabel("Amplitude")
    # plt.show()
    return ecg_segments

def normalize_ecg(ecg_data):
    #归一化处理
    # 将数据类型转换为float，以防万一
    nomal_Data=[]
    for ecg_data0 in ecg_data:
        ecg_data0=np.array(ecg_data0,dtype=float)
        # 计算最大值和最小值
        max_val = np.max(ecg_data0)
        min_val = np.min(ecg_data0)

        # 归一化处理
        m=(ecg_data0 - min_val) / (max_val - min_val)
        m=m.tolist()
        nomal_Data.append(m)
    # plt.plot(nomal_Data[0])
    # plt.title("ECG-nomaliaze-Data")
    # plt.show()

    return nomal_Data
def Cut_ST(heartbeats):
    After_Cut=[]
    Tmp_list=[]
    After_Cut_Point=[]
    Heart_Rate=(7500/(len(heartbeats[1])))*4
    for heartbeat_periods in heartbeats:
        # 将列表转换为NumPy数组
        heartbeat_periods_array = np.array(heartbeat_periods)

        # 寻找波谷和波峰
        valleys, _ = find_peaks(-heartbeat_periods_array)
        peaks, _ = find_peaks(heartbeat_periods_array)
        # # 找到前两个幅值最大的波峰
        # sorted_peaks_indices = np.argsort(heartbeat_periods_array[peaks])[::-1][:2]
        # top_peaks_indices = peaks[sorted_peaks_indices]
        # top_peaks_indices=np.min(top_peaks_indices)#T波
        if len(valleys) > 0 and len(peaks) > 1:
            # 找到第一个波谷作为S点
            s_point_index = valleys[0]
            J_point_index=s_point_index+25
            if Heart_Rate<100:
                t_point_index=J_point_index+40
            elif 100<=Heart_Rate<110:
                t_point_index=J_point_index+36
            elif 110<=Heart_Rate<120:
                t_point_index=J_point_index+32
            elif 120<=Heart_Rate<140:
                t_point_index=J_point_index+30
            else:
                t_point_index=J_point_index+20


            # 裁切出ST段
            st_segment = heartbeat_periods_array[J_point_index:t_point_index]
            Tmp_list=[J_point_index,t_point_index]
            After_Cut_Point.append(Tmp_list)
            After_Cut.append(st_segment)

    # # 可以打印或进一步处理裁切后的ST段的列表 st_segments
    # # 绘制心电图
    # t=np.arange(0,1000,1)
    # heartbeats[0]
    # plt.plot(t[:len(heartbeats[0])], heartbeats[0], label='Once-Heart-Beat-Data')
    #
    # # 在心电图中标记已知的两点
    # plt.scatter([After_Cut_Point[0][0], After_Cut_Point[0][1]],
    #             [heartbeats[0][After_Cut_Point[0][0]], heartbeats[0][After_Cut_Point[0][1]]], color='red')
    #
    # # 在两点位置画竖线
    # plt.axvline(After_Cut_Point[0][0], color='green', linestyle='--', label='J')
    # plt.axvline(After_Cut_Point[0][1], color='blue', linestyle='--', label='ST-End')
    # # 设置图例
    # plt.legend()
    #
    # # 显示图形
    # plt.show()
    # #print("裁切后的ST段列表:", After_Cut)
    # plt.plot(After_Cut[0])
    # plt.title("ECG-After_Cut-ST-Data")
    # plt.show()

    return After_Cut

def Delete_abnomal_Beat(heartbeats):
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

    return filtered_nomal
def detect_st_segment(ecg_signal, sampling_rate):
    # 使用差分法计算斜率
    derivative = np.diff(ecg_signal)

    # 设置一个阈值来检测ST段的斜率
    threshold = 0.1

    # 找到超过阈值的索引
    st_segment_indices = np.where(derivative > threshold)[0]

    return st_segment_indices
def Judge_cust(ecg_signals):
    new_ecg_signal=[]
    start_index=math.ceil(len(ecg_signals)/3)
    end_index=2*math.ceil(len(ecg_signals)/3)
    cust_signal=ecg_signals[start_index:end_index]
    cust_sign=max(cust_signal)
    former_signal=ecg_signals[:start_index]
    former_sign=max(former_signal)
    behind_signal=ecg_signals[end_index:]
    behind_sign=max(behind_signal)
    if former_sign>cust_sign*1.3 and behind_sign>cust_sign*1.3:
        new_ecg_signal=cust_signal
        print("有去除")
    elif former_sign>cust_sign*1.3:
        new_ecg_signal=ecg_signals[start_index:]
        print("有去除")
    else:
        new_ecg_signal=ecg_signals[:end_index]
        print("完整")


    return new_ecg_signal




def getData_l(folder_path):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    select_files=[file for file in files if file.endswith("L.mat") and file[:-5].isdigit()]
    if len(select_files)>25:
        select_files=select_files[:14]
        for file_name in select_files:
            file_path=os.path.join(folder_path,file_name)#拼成完整路径
            # 读取MATLAB .mat文件
            mat_data = scipy.io.loadmat(file_path)
            # 查看.mat文件中的变量信息
            mat_variables = scipy.io.whosmat(file_path)
            # print(mat_variables)
            # .mat文件中有一个名为 "filtered_ecg" 的变量，获取其中的第一行数据
            first_row_data = mat_data['filtered_ecg'][0, :]
        #     print(1)##
        # print(first_row_data)
        # plt.plot(first_row_data)
        # plt.title("ECG Data")
        # plt.xlabel("Sample")
        # plt.ylabel("Amplitude")
        # plt.show()
        return first_row_data
def getData(folder_path):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    select_files=[file for file in files if file.endswith(".mat") and file[:-4].isdigit()]
    if len(select_files)>30:
        select_files=select_files[:20]
        for file_name in select_files:
            file_path=os.path.join(folder_path,file_name)#拼成完整路径
            # 读取MATLAB .mat文件
            mat_data = scipy.io.loadmat(file_path)
            # 查看.mat文件中的变量信息
            mat_variables = scipy.io.whosmat(file_path)
            print(mat_variables)
            # .mat文件中有一个名为 "filtered_ecg" 的变量，获取其中的第一行数据
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
folder_path_l = r"D:\毕设数据\MyLowData"
folder_path=r"D:\毕设数据\数据\Train"
#getData(folder_path)
#获取数据
getData_l=getData_l(folder_path_l)
#判断是否只攫取中间1/3处心电图
Judge_cust=Judge_cust(getData_l)
#划分QRS波群
QRS_detect=QRS_detect_Blutty(Judge_cust,98)
#删去长度异常的QRS波群
Delete_abnomal_Beat=Delete_abnomal_Beat(QRS_detect)
#归一化处理
normalize_ecg=normalize_ecg(Delete_abnomal_Beat)
#低通滤波
L_filter_array=L_filter_array(normalize_ecg,20,500)
#M_filter_array=M_filter_array(normalize_ecg,8,16,500)
Cut_ST=Cut_ST(L_filter_array)
#get_stableData=get_stableData(getData,1,0.5,2000)
#应用低通
# l_filter_ecg=l_filter_ecg(get_stableData,100,2000)
# print(2)
#应用小波去噪
#小波去噪
#remove_noise=Daubechies_remove_noise(get_stableData,'db6',4)
# #带通去噪
# remove_noise=m_filter_ecg(get_stableData,2000,5,40)
# print(l_filter_ecg)
# print(3)
# remove_baseline_drift=remove_baseline_drift(remove_noise)
# normalize_ecg=normalize_ecg(remove_baseline_drift)
# print(4)
#
# QRS_detect_Blutty(normalize_ecg,95)

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
