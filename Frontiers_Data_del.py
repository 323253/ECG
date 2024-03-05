import os
import sys

import xml.etree.ElementTree as ET
import numpy as np
import math
import glob


def parse_xml_dir(filepaths, cut_size, step_pix=1000):
    """
    解析心电图（ECG）的XML文件，提取指定导联的数据，并切分为固定大小的样本。

    参数：
    - filepaths: 包含心电图XML文件路径的列表
    - cut_size: 指定样本的长度
    - step_pix: 切分步长，默认为1000

    返回：
    - all_leads: 包含所有样本的numpy数组
    """

    # 存储所有样本的列表
    all_leads = []

    # 遍历每个XML文件路径
    for xml_filepath in filepaths:

        # 解析XML文件
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        leads_12 = []

        # 标签前缀，用于解析XML中的命名空间
        tag_pre = ''
        for child in root:
            if child.tag.startswith('{urn:hl7-org:v3}'):
                tag_pre = '{urn:hl7-org:v3}'

        # 遍历XML文件中的元素
        for elem in tree.iterfind('./%scomponent/%sseries/%scomponent/%ssequenceSet/%scomponent/%ssequence' % (
        tag_pre, tag_pre, tag_pre, tag_pre, tag_pre, tag_pre)):
            for child_of_elem in elem:

                # 根据条件筛选导联数据
                if child_of_elem.tag == '%scode' % (tag_pre):

                    if child_of_elem.attrib['code'] == 'TIME_ABSOLUTE': break

                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V3R': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V4R': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V5R': break

                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V7': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V8': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V9': break

                    # 提取导联数据
                    for grand_child_digits in elem.iterfind('%svalue/%sdigits' % (tag_pre, tag_pre)):
                        arr = grand_child_digits.text.split(' ')
                        num_samples = np.array(arr).shape[0]
                        leads_12.append(arr)

        # 将导联数据转为NumPy数组
        leads_12 = np.array(leads_12)  # (12, 31600)
        leads_12_dim = leads_12.transpose(1, 0)  # (31600, 12)

        # 切分数据并进行标准化
        for idx in range(0, leads_12.shape[-1] - cut_size, step_pix):

            sample = leads_12_dim[idx:idx + cut_size, :]
            sample = sample.astype(np.float32)
            mean = np.mean(sample)
            std = np.std(sample)

            # 对样本进行标准化处理
            if std > 0:
                ret = (sample - mean) / std
            else:
                ret = sample * 0
            all_leads.append(ret)

            # 将样本列表转为NumPy数组并返回
    all_leads = np.array(all_leads)
    return all_leads
