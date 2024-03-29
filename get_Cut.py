import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import os
import pywt
from scipy.signal import find_peaks
import math

def QRS_detect_Blutty(ecg_signals, tp,file_name):
    ecg_segmentd=[]
    #分导联
    for i, ecg_signal in enumerate(ecg_signals):
        #防止零或者全为负值的情况
        if len(ecg_signal)>0 and np.any(ecg_signal)>0:
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
        else:
            ecg_segmentd.append([])
            print(file_name,f"第{i}导联失效")
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
            #print("前后去除")
        elif former_sign>cust_sign*1.3:
            new_ecg_signal.append(ecg_signals[start_index:])
            #print("前端去除")
        elif behind_sign > cust_sign * 1.3:
            new_ecg_signal.append(ecg_signals[:end_index])
            #print("后端去除")
        else:
            new_ecg_signal.append(ecg_signals)
            #print("完整")


    return new_ecg_signal
# 删除子列表中数字个数超过110个的元素
def Delete_longabnoemal_Data(ecg:list):
    for ecg_child in ecg:
        # 找到长度少于1010的数组元素的索引
        indices_to_remove = [i for i, arr in enumerate(ecg_child) if len(arr) >1053 or len(arr)<250]

        # 使用索引删除元素
        for index in reversed(indices_to_remove):
            ecg_child.pop(index)
        #print(len(indices_to_remove),"个心跳周期不具备参考价值 删除")

    return ecg
#利用正态分布去除长度异常心跳节拍
def Delete_abnomal_Beat(heartbeatd):
    filtered_nomals=[]
    for heartbeats in heartbeatd:
        # 计算每个元素的长度
        lengths = [len(item) for item in heartbeats]
        if len(lengths)>0:
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
#归一化处理
def normalize_ecg(ecg_datad):
    #归一化处理
    # 将数据类型转换为float，以防万一
    nomal_Datas=[]
    #分导联
    for ecg_data in ecg_datad:
        nomal_Data = []
        if len(ecg_data)>0:
            for ecg_data0 in ecg_data:
                ecg_data0=np.array(ecg_data0,dtype=float)
                # 计算最大值和最小值
                max_val = np.max(ecg_data0)
                min_val = np.min(ecg_data0)

                # 归一化处理
                m=(ecg_data0 - min_val) / (max_val - min_val)
                m=m.tolist()
                nomal_Data.append(m)
            nomal_Datas.append(nomal_Data)
        else:nomal_Datas.append([])

    return nomal_Datas
#切割出ST段
def Cut_ST(heartbeatd):
    After_Cuts=[]
    lable_STd=[]
    #分导联
    for heartbeats in heartbeatd:
        if heartbeats and len(heartbeats[0])>0:
            Heart_Rate = (7500 / (len(heartbeats[0]))) * 4
            After_Cut = []
            lable_STs=[]
            #一个导联中的N个心跳周期
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
                    # if Heart_Rate<100:
                    #     t_point_index=J_point_index+40
                    # elif 100<=Heart_Rate<110:
                    #     t_point_index=J_point_index+36
                    # elif 110<=Heart_Rate<120:
                    #     t_point_index=J_point_index+32
                    # elif 120<=Heart_Rate<140:
                    #     t_point_index=J_point_index+30
                    # else:
                    #     #t_point_index=J_point_index+20
                    #     continue
                    t_point_index = J_point_index + 40
                    # 裁切出ST段
                    st_segment = heartbeat_periods_array[J_point_index:t_point_index]
                    # Tmp_list=[J_point_index,t_point_index]
                    # After_Cut_Point.append(Tmp_list)
                    lable_ST = [1 if J_point_index <= i <= t_point_index else 0 for i in range(len(heartbeat_periods))]
                    lable_STs.append(lable_ST)
                    After_Cut.append(st_segment)
            After_Cuts.append(After_Cut)
            lable_STd.append(lable_STs)
        else:
            After_Cuts.append([])
            lable_STd.append([])
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

    return After_Cuts,lable_STd
#新的心跳与ST段储存在新的位置
def SaveToNewPath(ecg,save_folder_path,origenal_name):
    # 指定保存文件夹路径
    #save_folder_path = r"D:\毕设数据\HeartBeats"

    # 循环遍历 ecg 列表
    for i, sub_list in enumerate(ecg):
        # 构建新文件的完整路径
        new_file_name = f'{origenal_name}-{i+1}.mat'  # 以子列表的索引文件
        new_file_path = os.path.join(save_folder_path,new_file_name)
        #删除空列表元素
        filter_list=[item for item in sub_list if len(item)!=0]
        sub_list_array=np.array(filter_list)
        # 将子列表的数据保存到新的 .mat 文件
        scipy.io.savemat(new_file_path, {'variable_name': sub_list_array})

def Standard_Data(ecgtd):
    stand_datad=[]
    #分导联
    for ecgs in ecgtd:
        if len(ecgs)>0:
            #分每个ST段
            stand_datas=[]
            for ecg in ecgs:
                mean_value=np.mean(ecg)
                std_value=np.std(ecg)
                ecg_standard=(ecg-mean_value)/std_value
                stand_datas.append(ecg_standard)
            stand_datad.append(stand_datas)
        else:
            stand_datad.append([])
    return stand_datad

def getData_l(folder_path):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    select_files=[file for file in files if file.endswith("L.mat") and file[:-5].isdigit()]
    # 对文件名按照数字进行排序
    select_files.sort(key=lambda x: int(x[:-5]))
    if len(select_files)>1007:
        select_files=select_files[:1000]
        for file_name in select_files:
            print(file_name)
            file_path=os.path.join(folder_path,file_name)#拼成完整路径
            first_row_data = []
            # 读取MATLAB .mat文件
            mat_data = scipy.io.loadmat(file_path)

            rows=mat_data['filtered_ecg']
            for row in rows:
                first_row_data.append(row)
            #print(1)
            Judgecust = Judge_cust(first_row_data)
            QRS_detect = QRS_detect_Blutty(Judgecust, 98,file_name)
            Delete_longabnoemal=Delete_longabnoemal_Data(QRS_detect)
            normalize_ecgl=normalize_ecg(Delete_longabnoemal)
            Cut_STD,ST_Lable=Cut_ST(normalize_ecgl)
            Standard_D=Standard_Data(Cut_STD)
            #保存心跳周期
            # SaveToNewPath(normalize_ecgl,r"D:\毕设数据\HeartBeats",file_name[:-4])
            # print("A")
            #保存ST段
            SaveToNewPath(Standard_D,r"D:\毕设数据\ST_Select_Data_New",file_name[:-4])
            print('B')
            # SaveToNewPath(ST_Lable, r"D:\毕设数据\ST_Lable", file_name[:-4])
            # print('C')
        return first_row_data

# 指定文件夹路径
folder_path_l = r"D:\毕设数据\MyLowData"
folder_path=r"D:\毕设数据\数据\Train"
#getData(folder_path)
getData_l=getData_l(folder_path_l)