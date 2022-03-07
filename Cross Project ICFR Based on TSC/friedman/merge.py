#!/usr/bin/env python3

import fnmatch
import os
import sys
from pandas import read_excel, DataFrame
from scipy.io import arff
from typing import Iterable, Tuple


def new_path(path: str, ext='') -> str:
    'Return the original path, but with a different suffix name.'
    path = list(os.path.splitext(path))
    path[1] = ext
    return ''.join(path)


# 专门供scoring.py文件使用
def scoring_path(path, ext):
    filename = os.path.split(path)[-1].replace(".xlsx", "")
    first_folder = "scoring"
    filename += ext
    # 如果当前文件夹不存在，直接创建
    if not os.path.exists(first_folder):
        os.makedirs(first_folder)
    new_path = os.path.join(first_folder, filename)
    return new_path


def read_arff(path: str) -> DataFrame:
    return DataFrame(arff.loadarff(path)[0])


def load_pair(excel_path: str) -> Tuple[DataFrame, DataFrame]:
    arff_path = new_path(excel_path, '--CK_NET_PROC.arff')
    return read_excel(excel_path), read_arff(arff_path)


def feature_select(merged: DataFrame) -> Iterable[Tuple[str, DataFrame]]:
    groups = 'Clo dwReach'
    for group in groups.split():
        merged[group] = sum(
            merged[k] for k in merged.columns
            if k.endswith(group)
        )

    target_metrics = {
        'CK': 'loc2 wmc dit noc cbo rfc lcom',
        # ' ca ce npm lcom3 dam moa mfa cam ic cbm amc max_cc avg_cc'
        'NET': '*Ego Eff* Constraint Hierarchy Deg* Eigen* Between* ' + groups,
        'PROC': 'revision_num author_num lines* codechurn_*',
    }
    for name, metrics in target_metrics.items():
        target = DataFrame()
        metrics += ' bug loc'
        for metric_pattern in metrics.split():
            # 实现列表特殊字符的过滤或筛选,返回符合匹配模式（metric_pattern）的字符列表
            metric_group = fnmatch.filter(merged.columns, metric_pattern)
            if not metric_group:
                break
            for metric in metric_group:
                target[metric] = merged[metric]
        else:
            yield name, target


def main(excel_path: str):
    excel, arff = load_pair(excel_path)
    excel['loc2'] = excel['loc']
    arff['bug'] = arff.pop('isBug').map(lambda i: int(i == b'YES'))
    assert len(excel) == len(arff), '%s: %d %d -> %d' % (excel_path, len(excel), len(arff))
    for dataset in [excel, arff]:
        for name, df in feature_select(dataset):
            csv_path = new_path(excel_path, '-%s.csv' % name)
            df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        main(path)
