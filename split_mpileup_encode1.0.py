"""
Pileup encoder version 3.0
the first two version missing a serial sites that has no truth coverage but a SR coverage
the program ignore these sites and will cause futile preprocessing
so add the SR counts to be the label
start with the calculation_positions()
"""
import argparse

import numpy as np

import pandas as pd
import h5py
import os

mpileup_folder = "/itmslhppc/itmsl0212/Projects/python/wrsCode/VariantWorks/data/wrs_sample/split_mpilup/split_mpileup"
output_dir = "/itmslhppc/itmsl0212/Projects/python/wrsCode/VariantWorks/data/wrs_sample/split_mpilup/output"
feature_size = 10

label_symbols = ["*", "A", "C", "G", "T"]
feature_symbols = ["a", "c", "g", "t", "A", "C", "G", "T", "#", "*"]


def find_insertions(base_pileup):
    """Finds all of the insertions in a given base's pileup string.

    Args:
        base_pileup: Single base's pileup string output from samtools mpileup
    Returns:
        insertions: list of all insertions in pileup string
        next_to_del: whether insertion is next to deletion symbol (should be ignored)
    """
    insertions = []
    idx = 0
    next_to_del = []
    while idx < len(base_pileup):
        if base_pileup[idx] == "+" and base_pileup[idx+1].isdigit():
            end_of_number = False
            insertion_bases_start_idx = idx+1
            while not end_of_number:
                if base_pileup[insertion_bases_start_idx].isdigit():
                    insertion_bases_start_idx += 1
                else:
                    end_of_number = True
            insertion_length = int(base_pileup[idx:insertion_bases_start_idx])
            inserted_bases = base_pileup[insertion_bases_start_idx:insertion_bases_start_idx+insertion_length]
            insertions.append(inserted_bases)
            next_to_del.append(True if base_pileup[idx - 1] in '*#' else False)
            idx = insertion_bases_start_idx + insertion_length + 1  # skip the consecutive base after insertion
        else:
            idx += 1
    return insertions, next_to_del


def calculate_positions(start_pos, end_pos, subreads, sr_coverage, truth_coverage):
    """Calculates positions array from read pileup columns.

    Args:
        start_pos: Starting index of pileup columnsyjf
        end_pos: Ending index of pileup columns
        subreads: Array of subread strings from pileup file
        truth_coverage: Array of integers specifying coverage at each pileup column
        exclude_no_coverage_positions: Boolean specifying whether to include 0 coverage
        positions during position calculation
    Returns:
        positions: Array of tuples containing major/minor positions of pileup
    """
    positions = []
    # Calculate ref and insert positions

    for i in range(start_pos, end_pos):

        if truth_coverage[i] == 0 and sr_coverage[i] == 0:
            lr_pos = []
            lr_pos.append([i, 0])
            positions += lr_pos  # 双空白位点给自己占一个位子先 wrs code
            continue
        else:
            base_pileup = subreads[i].strip("^]").strip("$")

            # Get all insertions in pileup
            insertions, next_to_del = find_insertions(base_pileup)

            # Find length of maximum insertion
            longest_insertion = len(max(insertions, key=len)) if insertions else 0

            # Keep track of ref and insert positions in the pileup and the insertions
            # in the pileup.
            ref_insert_pos = []  # ref position for ref base pos in pileup, insert for additional inserted bases
            ref_insert_pos.append([i, 0])
            for j in range(longest_insertion):
                ref_insert_pos.append([i, j + 1])
            positions += ref_insert_pos

    return positions

def reencode_base_pileup(ref_base, pileup_str):
    """Re-encodes mpileup output of list of special characters to list of nucleotides.

    Args:
        ref_base : Reference nucleotide
        pileup_str : mpileup encoding of pileup bases relative to reference
    Returns:
        A pileup string of special characters replaced with nucleotides.
    """
    pileup = []
    for c in pileup_str:
        if c == "." or c == "*":  # wrs code or c == "*" 只要没有的一律拿长读本身代替
            pileup.append(ref_base)
        elif c == ",":
            pileup.append(ref_base.lower())
        else:
            pileup.append(c)
    return "".join(pileup)


def feature_encode(pileup):
    """
    feature generate for each seq in LR set with the information in SR reads which aligned to it.
    :param pileup:
    :return:
    """
    start_pos = 0
    end_pos = len(pileup)
    subreads = pileup[:, 4]
    sr_coverage = pileup[:,3].astype("int")
    truth_coverage = pileup[:, 7].astype("int")
    positions = calculate_positions(start_pos, end_pos, subreads, sr_coverage, truth_coverage)
    positions = np.array(positions)


    num_features = len(feature_symbols)

    pileup_counts = np.zeros(shape=(len(positions), num_features + 2), dtype=np.float32) # wrs + 2
    ref_info = None
    for i in range(len(positions)):
        ref_position = positions[i][0]
        insert_position = positions[i][1]
        base_pileup = subreads[ref_position].strip("^]").strip("$")
        base_pileup = reencode_base_pileup(pileup[ref_position, 2], base_pileup)  #  改了reencode函数，使feature从长读本身来
        insertions, next_to_del = find_insertions(base_pileup)
        insertions_to_keep = []

        pileup_counts[i, 10] = truth_coverage[ref_position]
        pileup_counts[i, 11] = sr_coverage[ref_position]

        # Remove all insertions which are next to delete positions in pileup
        for k in range(len(insertions)):
            if next_to_del[k] is False:
                insertions_to_keep.append(insertions[k])

        # Replace all occurrences of insertions from the pileup string
        for insertion in insertions:
            base_pileup = base_pileup.replace("+" + str(len(insertion)) + insertion, "")

        if insert_position == 0:  # No insertions for this position
            for j in range(len(feature_symbols)):
                pileup_counts[i, j] = base_pileup.count(feature_symbols[j])
            # Add draft base and base quality to encoding

        elif insert_position > 0:
            # Remove all insertions which are smaller than minor position being considered
            # so we only count inserted bases at positions longer than the minor position
            insertions_minor = [x for x in insertions_to_keep if len(x) >= insert_position]
            for j in range(len(insertions_minor)):
                inserted_base = insertions_minor[j][insert_position - 1]
                pileup_counts[i, feature_symbols.index(inserted_base)] += 1

    return pileup_counts, positions

def label_encode(pileup):
    """
    label encode for each seq with the pileup information from truth coverage
    :param pileup:
    :return:
    """
    start_pos = 0
    end_pos = len(pileup)
    subreads = pileup[:, 4]
    truth_coverage = pileup[:, 7].astype("int")
    sr_coverage = pileup[:,3].astype("int")
    positions = calculate_positions(start_pos, end_pos, subreads, sr_coverage, truth_coverage)
    positions = np.array(positions)


    truth = pileup[:, 8]
    labels = np.zeros((len(positions),))  # gap, A, C, G, T (sparse format)
    for i in range(len(positions)):
        reference_pos = positions[i][0]
        inserted_pos = positions[i][1]
        truth_base = truth[reference_pos].strip("^]").strip("<").strip("$").upper()   #wrs code 加了.strip("<").strip(",")
        truth_base = reencode_base_pileup(pileup[reference_pos, 2], truth_base)
        # Handle minor position label (no insertion)
        if (inserted_pos == 0):
            if ("+" in truth_base):
                ref_base = truth_base.split("+")[0]  # 从pileup字符串中解析出来的字符编成标签,先判断是不是在label规范里，再大写，或者忽略
                if ref_base.upper() not in label_symbols: # 大写的ref 不在symbol 列表里，则label直接置0，在列表中才编码
                    labels[i] = 0.
                else:
                    labels[i] = label_symbols.index(ref_base.upper())
            elif ("-" in truth_base):
                ref_base = truth_base.split("-")[0].upper()
                if ref_base.upper() not in label_symbols: # 大写的ref 不在symbol 列表里，则label直接置0，在列表中才编码
                    labels[i] = 0.
                else:
                    labels[i] = label_symbols.index(ref_base.upper())
            elif (truth_base in label_symbols):
                labels[i] = label_symbols.index(truth_base)
        # Handle major position label (with insertion)
        elif (inserted_pos > 0):
            if ("+" in truth_base):
                inserted_bases = truth_base.split("+")[1].upper()
                inserted_bases = "".join(
                    [i for i in inserted_bases if i.isdigit() is False])
                if (len(inserted_bases) >= inserted_pos):
                    inserted_truth_base = inserted_bases[inserted_pos - 1].upper()
                    labels[i] = label_symbols.index(inserted_truth_base)
        else:
            raise RuntimeError(
                "Encode labels error - inserted position should be >= 0.")

    return labels, positions

def write_wrs(id,read,file):
    with open(file, mode='a') as f:
        seq_name = ">" + id + "\n"
        f.write(seq_name)
        f.write(read)
        f.write("\n")
    f.close()


def pack_feature_encode(mpileup_file):
    """
    split the pileup for per seq and pack it together again，
    then encode per seq_region separately and add the seq number info to each region,
    finally pack the nd.array together and return
    :return:
    """
    pileup_total = pd.read_csv(mpileup_file, delimiter="\t", header=None, quoting=3).values
    seq_name = pileup_total[:, 0]  # the first column of pileup is the seq_name set
    sequence = pileup_total[:, 2]
    truth_cov = pileup_total[:, 7]
    sr_cov = pileup_total[:, 3]

    pileup_counts = np.empty(shape=(0, feature_size+2), dtype=np.float32)
    positions = np.empty(shape=(0, 3), dtype=np.int64)
    labels = np.empty(shape=(0,), dtype=np.int64)

    cur_start = 0
    seq_num = 1  # 序列编号从1开始

    for i in range(0, len(pileup_total) - 1):
        if seq_name[i] == seq_name[i + 1]:
            continue

        truth = truth_cov[cur_start: i+1]
        sr = sr_cov[cur_start: i+1]
        if not (np.any(truth)) and not (np.any(sr)):  # 如果全0则啥也不干
            continue
            # cur_end = i + 1
            # read = ''.join([e for e in sequence[cur_start: cur_end]])
            # print("uncovered: ", seq_name[i])
            # print(read)
            # write_wrs(seq_name[i], read)  # 将序列写出去
            # read = ''
        else:
            cur_end = i + 1
            print("covered: ", seq_name[i])
            pileup = pileup_total[cur_start:cur_end, :]  # split pileup into per seq

            feature, feature_position = feature_encode(pileup)  # region for per seq

            if len(feature) == 0 or len(feature_position) == 0:  # if the seq has no coverage from SR or truth,ignore it
                cur_start = i + 1
                seq_num += 1  # seq_num解码的时候说不定还可以拿去还原序列号呢，所以先自加1
                continue

            feature_position = np.insert(feature_position, [0], seq_num, axis=1)

            label, label_position = label_encode(pileup)
            assert len(feature) == len(label)

            label_position = np.insert(label_position, [0], seq_num, axis=1)
            pileup_counts = np.append(pileup_counts, feature, axis=0) # 拼接
            positions = np.append(positions, feature_position, axis=0)
            labels = np.append(labels, label, axis=0)

        cur_start = cur_end
        seq_num += 1


    assert len(positions) == len(pileup_counts) or len(positions) == len(labels)

    print("position array are equal")

    return pileup_counts, positions, labels


def sliding_window(array, window, step=1, axis=0):
    """Generate chunks for encoding and labels.

    Args:
        array: Numpy array with the pileup counts
        window: Length of output chunks
        step: window minus chunk overlap
        axis: defaults to 0
    Returns:
        Iterator with chunks
    """
    chunk = [slice(None)] * array.ndim
    end = 0
    chunks = []
    for start in range(0, array.shape[axis] - window + 1, step):
        end = start + window
        chunk[axis] = slice(start, end)
        chunks.append(array[tuple(chunk)])
    if array.shape[axis] > end:
        start = array.shape[axis] - window
        chunk[axis] = slice(start, array.shape[axis])
        chunks.append(array[tuple(chunk)])
    return chunks


def encode(chunk_len, chunk_ovlp, input_file, out_file):  # chunk_len=1000, chunk_ovlp=200
    """
    the function inculde three parts: encode the feature and label,
    count the position according the alignment file
    slice the feature_encode and take account the overlaps=200
    next is feeding into the network
    :return:
    """

    encoding, encoding_positions, label = pack_feature_encode(input_file)  # generate the temp pileup data and position from the aligned data
    read_ids = encoding_positions[:, 0]
    assert (len(encoding) == len(label)), print("Encoding and label dimensions not as expected:", )
    assert (len(encoding_positions) == len(encoding)), print("Encoding and positions not as expected:", )

    encoding_chunks = sliding_window(encoding, chunk_len, step=chunk_len - chunk_ovlp)
    position_chunks = sliding_window(encoding_positions, chunk_len, step=chunk_len - chunk_ovlp)
    label_chunks = sliding_window(label, chunk_len, step=chunk_len - chunk_ovlp)
    read_id_chunks = sliding_window(read_ids, chunk_len, step=chunk_len - chunk_ovlp)
    # output data
    features = []  # features in column
    labels = []  # correct labeling
    positions = []  # track match/insert for stitching
    read_ids = []  # track folder name and windows
    features += (encoding_chunks)
    labels += (label_chunks)
    positions += (position_chunks)
    read_ids += (read_id_chunks)
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)
    positions = np.stack(positions, axis=0)
    h5_file = h5py.File(out_file, 'w')
    h5_file.create_dataset('features', data=features)
    h5_file.create_dataset('positions', data=positions)
    h5_file.create_dataset('labels', data=labels)
    h5_file.create_dataset('read_ids', data=read_ids)
    h5_file.close()
    # return encoding_chunks, position_chunks, label_chunks, read_id_chunks

    return [], [], [], []


def check_file(file):
    if os.path.exists(file):
        print(file, " is exists!")
        os.remove(file)

def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Encoder the split mpileup files into split hdf file.')
    parser.add_argument('--mpileup-folder', type=str, help='The path of split_mpileup foder')
    parser.add_argument('--output-folder', type=str, help='Path to output HDF5 file.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """
    single process to encode per split mpileup into hdf file
    """
    parsed_args = build_parser()
    for i in range(1, len(os.listdir(parsed_args.mpileup_folder)) + 1):
        pileup_file = os.path.join(parsed_args.mpileup_folder, "mpileup_genome_{}.pileup".format(i))
        hdf_file = os.path.join(parsed_args.output_folder, "mpileup_genome_{}.hdf".format(i))
        #uncovered_file = os.path.join(parsed_args.output_folder, "mpileup_genome_{}uc.fasta".format(i))
        #check_file(uncovered_file)  # clear the output file to avoid overlying
        encode(chunk_len=1000, chunk_ovlp=200, input_file=pileup_file, out_file=hdf_file)
        print("finish the {}/{} pileup encode:".format(i, len(os.listdir(parsed_args.mpileup_folder))))


