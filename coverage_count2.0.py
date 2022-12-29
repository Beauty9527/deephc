"""
count the coverage of long reads version 2.0
count the coverage states of pileup, and try to make it in multi-processes

"""


import os
import pandas as pd
import numpy as np
import h5py

from functools import partial
import multiprocessing as mp

def get_file_list(file_path):
    fileList = []
    fileName = []
    for root, dirs, files in os.walk(file_path):

        for fileObj in files:
            # 空列表写入遍历的文件名称，目录路径拼接文件名称
            fileName.append(fileObj)
            path = os.path.join(root, fileObj).replace('\\', '/')
            fileList.append(path)
        return fileList

def count_coverage(mpileup_file):
    pileup = pd.read_csv(mpileup_file, delimiter="\t", header=None, quoting=3).values # read it first
    sr_coverage = pileup[:,3].astype("int")
    truth_coverage = pileup[:, 7].astype("int")
    count_11 = 0
    count_01 = 0
    count_10 = 0
    count_00 = 0
    for i in range(len(pileup)):
        if sr_coverage[i] != 0 and truth_coverage[i] != 0:
            count_11 += 1
        if sr_coverage[i] == 0 and truth_coverage[i] != 0:
            count_01 += 1
        if sr_coverage[i] != 0 and truth_coverage[i] == 0:
            count_10 += 1
        if sr_coverage[i] == 0 and truth_coverage[i] == 0:
            count_00 += 1

    # count_11 /= len(pileup)
    # count_01 /= len(pileup)
    # count_10 /= len(pileup)
    # count_00 /= len(pileup)
    # print("The rate of count_11: ", (count_11))
    return count_11, count_01, count_10, count_00, len(pileup)



def collect_coverage():

    count_coverages = partial(count_coverage) # 固定一部分参数

    # output data
    counts_11 = 0
    counts_01 = 0
    counts_10 = 0
    counts_00 = 0
    len_pileups = 0

    label_idx = 0
    pool = mp.Pool()
    for out in pool.imap(count_coverages, file_list):

        (count_11, count_01, count_10, count_00, len_pileup) = out
        counts_11 += count_11
        counts_01 += count_01
        counts_10 += count_10
        counts_00 += count_00
        len_pileups += len_pileup
        print("Generated {} pileup files".format(label_idx))
        label_idx += 1
    pool.close()
    print("The rate of doubleST: ", (counts_11 / len_pileups))
    print("The rate of onlyT: ", (counts_01 / len_pileups))
    print("The rate of onlyS: ", (counts_10 / len_pileups))
    print("The rate of noST: ", (counts_00 / len_pileups))


if __name__ == '__main__':

    folders_to_encode = "/itmslhppc/itmsl0212/Projects/python/wrsCode/VariantWorks/data/validation/yeast/pacbio/split_mpileup"
    file_list = get_file_list(folders_to_encode)
    # mpilup_file = "/itmslhppc/itmsl0212/Projects/python/wrsCode/VariantWorks/data/wrs_sample/fan/mpileup_genome.pileup"
    # count_coverage(mpilup_file)
    collect_coverage()