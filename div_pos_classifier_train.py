# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:02:31 2018

@author: jeffrey_yuan
"""

import time, argparse, utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pickle
from sklearn.externals import joblib
import os

def main():
    start_time = time.time()

    int_pos_folder = os.path.join("..", "kmer_position_confirmation", 
        "nctc9964.repeat.5.forward.integrated_pos.with.polished.cons.2.as.true.output")
        
    freq_file = os.path.join(int_pos_folder, 
                             "divergence_frequencies.txt")
    pk_out_file = os.path.join(int_pos_folder, 
                               "pos_to_kmer_output.txt")
    pos_out_file = os.path.join(int_pos_folder, 
                                "all_pos_output.txt")
    freq_outs = utils.read_div_freq_file(freq_file)
    pk_out = read_pk_output(pk_out_file)
    pk_pos, pk_div, pk_counts, pk_kmers = pk_out
    pos_out = read_pos_output(pos_out_file)
    
    k = 10
    freq_width = 4
    kmer_width = 2 * k
    
    freq_features = np.zeros(shape = (len(freq_outs), freq_width))
    kmer_features = np.zeros(shape = (len(freq_outs), kmer_width))
    for i, freq in enumerate(freq_outs):
        pos = freq[0]
        cov = freq[1]
        sub_ct = freq[5]
        del_ct = freq[7]
        ins_ct = freq[9]
        pkc = pk_counts[i][:2 * k]
        norm_pkc = [float(c) / cov for c in pkc]
        freq_fts = [cov, sub_ct, del_ct, ins_ct]
        freq_features[i][0:freq_width] = freq_fts
        kmer_features[i][0:len(norm_pkc)] = norm_pkc
    
    true_labels = np.array([-1 for _ in pos_out])
    for i, out in enumerate(pos_out):
        true = out[5].strip()
        if true == "YES":
            true_labels[i] = 1
        else:
            true_labels[i] = 0
        
    labels = ["cov", "sub", "del", "ins", 
              "kmer_pca1", "kmer_pca2", "kmer_pca3", "kmer_pca4", "kmer_pca5"]
    large_indels = [(0, 2000), (19610, 21050), (21865, 23975), (31860, 33185)]
    non_indel_inds = [(2000, 19610), (21050, 21865), (23975, 31860), (33185, )]
    freq_feat_arr = np.concatenate((freq_features[2000:19610], 
                                   freq_features[21050:21865], 
                                   freq_features[23975:31860], 
                                   freq_features[33185:]), axis = 0)
    kmer_feat_arr = np.concatenate((kmer_features[2000:19610], 
                                   kmer_features[21050:21865], 
                                   kmer_features[23975:31860], 
                                   kmer_features[33185:]), axis = 0)
    lab_arr = np.concatenate((true_labels[2000:19610], 
                              true_labels[21050:21865], 
                              true_labels[23975:31860], 
                              true_labels[33185:]))
    print "all freq features", freq_features.shape
    print "all kmer features", kmer_features.shape
    print "all true labels", true_labels.shape
    print "filtered freq features", freq_feat_arr.shape
    print "filtered kmer features", kmer_feat_arr.shape
    print "filtered true labels", lab_arr.shape
    
    all_feat_arr = np.concatenate((freq_feat_arr, kmer_feat_arr), axis = 1)
    print "concatenated features", all_feat_arr.shape
    
    orig_x_train, orig_x_test, y_train, y_test = \
        train_test_split(all_feat_arr, lab_arr, test_size = 0.25, 
                         random_state = 0)
    
    feat_tot = freq_width + kmer_width
    split_train = np.array_split(orig_x_train, [freq_width], 
                                 axis = 1)
    x_freq_train = split_train[0]
    x_kmer_train = split_train[1]
    print "X freq", x_freq_train.shape
    print "X kmer", x_kmer_train.shape
    print
    split_test = np.array_split(orig_x_test, [freq_width], 
                                 axis = 1)
    x_freq_test = split_test[0]
    x_kmer_test = split_test[1]
            
    num_pca_components = 5
    pca = PCA(n_components = num_pca_components)
    pca.fit(x_kmer_train)
    print "PCA explained variance", pca.explained_variance_ratio_
    print "PCA singular values", pca.singular_values_
    
    x_kmer_pca_train = pca.transform(x_kmer_train)
    x_kmer_pca_test = pca.transform(x_kmer_test)
    
    x_train_pca_concat = np.concatenate((x_freq_train, x_kmer_pca_train), 
                                        axis = 1)
    x_test_pca_concat = np.concatenate((x_freq_test, x_kmer_pca_test), 
                                       axis = 1)
    
    x_train = scale(x_train_pca_concat)
    x_test = scale(x_test_pca_concat)
    
    #logRegr = LogisticRegression(penalty = "l1", solver = "liblinear", 
    #                             multi_class = "auto")
    logRegr = LogisticRegressionCV(penalty = "l1", cv = 5, 
                                   solver = "liblinear", multi_class = "auto")
    logRegr.fit(x_train, y_train)
    coef = logRegr.coef_
    print "Coefficients"
    for l, c in zip(labels, coef[0]):
        print "{0}\t{1}".format(l, c)
    print
    
    train_predictions = logRegr.predict(x_train)
    train_probs = logRegr.predict_proba(x_train)
    train_probs_first = train_probs[:, 1]
    print "Train Score", logRegr.score(x_train, y_train)
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_probs_first)
    train_roc_auc = auc(train_fpr, train_tpr)
    train_conf_mat = confusion_matrix(y_train, train_predictions)
    print "Train Confusion Matrix"
    print "TN FP"
    print "FN TP"
    print train_conf_mat
    print
    
    test_predictions = logRegr.predict(x_test)
    test_probs = logRegr.predict_proba(x_test)
    test_probs_first = test_probs[:, 1]
    print "Test Score", logRegr.score(x_test, y_test)
    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, test_probs_first)
    test_roc_auc = auc(test_fpr, test_tpr)
    test_conf_mat = confusion_matrix(y_test, test_predictions)
    print "Test Confusion Matrix"
    print "TN FP"
    print "FN TP"
    print test_conf_mat
    print
    
    classifier_file = os.path.join(int_pos_folder, 
                                   "nctc9964.rep_5.log_reg_classifier.sav")
    
    #pickle.dump(logRegr, open(classifier_file, "wb"))
    joblib.dump(logRegr, classifier_file)
    
    print "Total time:\t{0:.3f} s".format(time.time() - start_time)

def read_pk_output(pk_out_file):
    pos = []
    div = []
    counts = []
    kmers = []
    with open(pk_out_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line and line[0] == ">":
                parts = line.split("_")
                pos.append(int(parts[0][1:]))
                div.append(parts[1])
            elif line:
                counts.append([])
                kmers.append([])
                for kmer_part in line.split(","):
                    count, kmer = kmer_part.split("-")
                    counts[-1].append(int(count))
                    kmers[-1].append(kmer)
    return pos, div, counts, kmers

def read_pos_output(pos_file):
    pos_out = []
    with open(pos_file, "r") as f:
        int_inds = [0, 8, 9]
        float_inds = [4, 7]
        for i, line in enumerate(f):
            line = line.strip()
            if i > 0:
                parts = line.split("\t")
                for ind in int_inds:
                    parts[ind] = int(round(float(parts[ind])))
                for ind in float_inds:
                    parts[ind] = float(parts[ind])
                pos_out.append(parts)
    return pos_out

if __name__ == "__main__":
    main()