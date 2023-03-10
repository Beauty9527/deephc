"""
tf_infer version 4.0
version 2.0 model.predict() generate a prob
3.0 model.predict_classes() generate the index of max prob use numpy to instead the flatten step,
4.0 take all split hdf into a folder,and infer them one by one because GPU memory OOM
numpy flatten is no use because the big data,change back to list flatten
"""
import argparse

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import data_generator
import os

def check_file(file):
    if os.path.exists(file):
        print(file, " is exists!")
        os.remove(file)

def flatten(all_preds,  all_read_id):
    flat_all_preds = []
    flat_all_read_id = []
    for sublist in all_preds:
        for item in sublist:
            flat_all_preds.append(item)
    for sublist in all_read_id:
        for item in sublist:
            flat_all_read_id.append(item)

    return flat_all_preds, flat_all_read_id


def ovlap_remv(pre_results, read_ids):
    """
    the difference between 3.0 and 4.0
    :param pre_results: the results from infer
    :param read_ids: the data from hdf file
    :return: falttened list
    """
    assert (len(pre_results) == len(read_ids))

    all_preds = []
    all_read_id = []
    for i in range(len(pre_results)):
        if i==0:
            # all_preds = np.append(all_preds,pre_results[i])
            # all_read_id = np.append(all_read_id, read_ids[i])
            all_preds.append(pre_results[i])
            all_read_id.append(read_ids[i])

        else:
            # all_preds = np.append(all_preds, pre_results[i][200:])
            # all_read_id = np.append(all_read_id, read_ids[i][200:])
            all_preds.append(pre_results[i][200:])
            all_read_id.append(read_ids[i][200:])

    f_all_preds, f_all_read_id = flatten(all_preds, all_read_id)

    return f_all_preds, f_all_read_id
            #all_preds.astype(int), all_read_id.astype(int)


def cut_probs_write(all_labels, read_id):
    assert (len(all_labels) == len(read_id))
    start = 0
    for i in range(len(read_id)-1):
        if read_id[i] == read_id[i+1]:
            continue
        else:
            cur_read_label = all_labels[start:i+1]
            read = decode_per_read(cur_read_label)
            write_wrs(str(read_id[start]), read, output_fasta)
            start = i + 1 # this unclear start cause the double maximum read length, because there are two sequence >1


def decode_per_read(cur_read_label):
    """Decode probabilities into sequence by choosing the Nucleotide base with the highest probability.

    Returns:
        seq: sequence output from probabilities
    """
    label_symbols = ["*", "A", "C", "G", "T"]  # Corresponding labels for each network output channel
    seq = ''
    for i in range(len(cur_read_label)):
        base = cur_read_label[i]
        nuc = label_symbols[base]
        if nuc != '*':
            seq += nuc
    return seq


def write_wrs(id, read, file):
    with open(file, mode='a') as f:
        seq_name = ">" + id + "\n"
        f.write(seq_name)
        f.write(read)
        f.write("\n")
    f.close()




def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Iter load the split_hdf file to infer and decode.')

    parser.add_argument('--hdf-folder', type=str, help='The splited mpileup folder.')
    parser.add_argument('--output-folder', type=str, help='Path to save the inferred fastas.')
    parser.add_argument('--model-hdf5-path', type=str, help='Path to save model.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parsed_args = build_parser()
    hdf_path = parsed_args.hdf_folder
    output_dir = parsed_args.output_folder
    model_hdf5_path = parsed_args.model_hdf5_path

    model = models.load_model(model_hdf5_path)

    for i in range(1, len(os.listdir(hdf_path)) + 1):  # len(os.listdir(hdf_path)) + 1
        hdf_file = os.path.join(hdf_path, "mpileup_genome_{}.hdf".format(i))
        output_fasta = os.path.join(output_dir, "mpileup_genome_{}.fasta".format(i))
        check_file(output_fasta)  # clear the output file to avoid overlying
        # predict_x= data_generator.load_hdf_to_predict(hdf_file, ["features"])
        # dataset_predict = tf.data.Dataset.from_tensor_slices(predict_x)
        # predict_data = dataset_predict.repeat(1).batch(256).prefetch(1)
        # load one time or load iter are suitable for split hdf generate from split mpileup 59

        predict_x, read_ids = data_generator.load_hdf_to_predict(hdf_file, ["features", "read_ids"])
        predict_data = data_generator.load_hdf_iter_predict(hdf_file, keys="features")  # instance the data generator
        results = np.argmax(model.predict(predict_data, batch_size=32, verbose=1), axis=-1)  #predict the reaults per batch

        all_labels, read_id = ovlap_remv(results, read_ids)
        cut_probs_write(all_labels, read_id)
        print("Finish the {} hdf infer and decode".format(i))


    print("Predict success, the results are in corrected_covered.fasta!")

