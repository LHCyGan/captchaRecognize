# -*- encoding:utf-8 -*-
# author: liuheng
import torch

__all__ = ['str_to_tensor', 'tensor_to_str', 'ctc_to_str', 'parse_label_map_c2i', 'parse_label_map_i2c']


def parse_label_map_c2i(label_map):
    """
    解析 LabelMap 为 Char -> Index
    :param label_map: 原始码表
    :return: Dict
    """
    if isinstance(label_map, dict):
        return label_map
    elif isinstance(label_map, list):
        return dict(zip(label_map, list(range(0, len(label_map)))))
    elif isinstance(label_map, str):
        return parse_label_map_c2i([i for i in label_map])
    else:
        raise TypeError("LabelMap must be dict list or str")

def parse_label_map_i2c(label_map):
    """
    解析 LabelMap 为 Index -> Char 的字典
    :param label_map: 原始码表
    :return: Dict
    """
    if isinstance(label_map, dict):
        return label_map
    elif isinstance(label_map, list):
        return dict(zip(list(range(0, len(label_map))), label_map))
    elif isinstance(label_map, str):
        return parse_label_map_i2c([i for i in label_map])
    else:
        raise TypeError("LabelMap must be dict list or str")

def str_to_tensor(label: str, label_map):
    """
    文本编码
    :param label: 标签文本
    :param label_map: 码表
    :return: 编码后的文本
    """
    label_map = parse_label_map_c2i(label_map)
    data = [label_map[c] for c in label]
    return torch.as_tensor(data).long()

def tensor_to_str(data, label_map):
    """
    文本编码
    :param data: 编码后的文本
    :param label_map: 码表
    :return:  解码后文本
    """
    label_map = parse_label_map_i2c(label_map)
    return ''.join([label_map[int(i)] for i in list(data)])

def ctc_to_str(data, label_map):
    """
    CTC 解码
    :param data: 编码后的文本
    :param label_map: 码表
    :return: 解码后的文本
    """
    result = []
    last = -1
    for i in list(data):
        if i == 0:
            last = -1
        elif i != last:
            result.append(i)
            last = i
    return tensor_to_str(result, label_map)