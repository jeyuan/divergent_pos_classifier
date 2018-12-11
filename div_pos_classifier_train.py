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
    
    k = 10
    freq_width = 4
    kmer_width = 2 * k
    feat_tot = freq_width + kmer_width            
    labels = ["cov", "sub", "del", "ins", 
              "kmer_pca1", "kmer_pca2", "kmer_pca3", "kmer_pca4", "kmer_pca5"]
    classifier_path = os.path.join("..", "kmer_position_confirmation", 
                "classifiers")
    clf_files = ["nctc9964.rep_5.log_reg_classifier.sav", 
                 "sim.A.20k.x.15.log_reg_classifier.sav", 
                 "nctc9964.rep_5.sim.A.20k.x.15.joint.classifier.sav"]
    classifier_file = os.path.join(classifier_path, 
                                   clf_files[2])
    
    nctc_labels = nctc9964_rep5_train(k, freq_width, kmer_width)
    nctc_freq, nctc_kmer, nctc_true = nctc_labels
    nctc_features = np.concatenate((nctc_freq, nctc_kmer), axis = 1)
    print "NCTC freq features", nctc_freq.shape
    print "NCTC kmer features", nctc_kmer.shape
    print "NCTC features", nctc_features.shape
    print "NCTC true labels", nctc_true.shape
    
    """sim_labels = sim_a_20k_x_15_train(k, freq_width, kmer_width)
    sim_freq, sim_kmer, sim_true = sim_labels
    sim_features = np.concatenate((sim_freq, sim_kmer), axis = 1)
    print "Sim freq features", sim_freq.shape
    print "Sim kmer features", sim_kmer.shape
    print "Sim features", sim_features.shape
    print "Sim true labels", sim_true.shape"""
    
    all_features = nctc_features
    all_true = nctc_true
    #all_features = np.concatenate((nctc_features, sim_features), axis = 0)
    #all_true = np.concatenate((nctc_true, sim_true), axis = 0)
    print "All features", all_features.shape
    print "All true labels", all_true.shape
    orig_x_train, orig_x_test, y_train, y_test = \
        train_test_split(all_features, all_true, test_size = 0.25, 
                         random_state = 0)
    
    split_train = np.array_split(orig_x_train, [freq_width], 
                                 axis = 1)
    x_freq_train = split_train[0]
    x_kmer_train = split_train[1]
    print "X freq train", x_freq_train.shape
    print "X kmer train", x_kmer_train.shape
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
    
    #clf = LogisticRegression(penalty = "l1", solver = "liblinear", 
    #                             multi_class = "auto")
    clf = LogisticRegressionCV(penalty = "l1", cv = 5, 
                                   solver = "liblinear", multi_class = "auto")
    clf.fit(x_train, y_train)
    coef = clf.coef_
    print "Coefficients"
    for l, c in zip(labels, coef[0]):
        print "{0}\t{1}".format(l, c)
    print
    
    train_out = make_predictions(clf, x_train, y_train)
    train_predictions, train_probs, train_roc_auc, train_conf_mat = train_out
    print "Train Score", clf.score(x_train, y_train)
    print "Train Confusion Matrix:"
    print "TN FP"
    print "FN TP"
    print train_conf_mat
    print
    
    test_out = make_predictions(clf, x_test, y_test)
    test_predictions, test_probs, test_roc_auc, test_conf_mat = test_out
    print "Test Score", clf.score(x_test, y_test)
    print "Test Confusion Matrix:"
    print "TN FP"
    print "FN TP"
    print test_conf_mat
    print
    
    #pickle.dump(clf, open(classifier_file, "wb"))
    joblib.dump(clf, classifier_file)
    
    print "Total time:\t{0:.3f} s".format(time.time() - start_time)
    

def nctc9964_rep5_train(k, freq_width, kmer_width):
    int_pos_folder = os.path.join("..", "kmer_position_confirmation", 
        "nctc9964.repeat.5.forward.integrated_pos.with.polished.cons.2.as.true.output")
        
    freq_file = os.path.join(int_pos_folder, 
                             "divergence_frequencies.txt")
    pk_out_file = os.path.join(int_pos_folder, 
                               "pos_to_kmer_output.txt")
    pos_out_file = os.path.join(int_pos_folder, 
                                "all_pos_output.txt")
    freq_out = utils.read_div_freq_file(freq_file)
    pk_out = read_pk_output(pk_out_file)
    pos_out = read_pos_output(pos_out_file)
    
    freq_features, kmer_features = get_features(freq_out, pk_out, k, 
                                                freq_width, kmer_width)
    true_labels = get_true_labels(pos_out)
    
    freq_feat_arr = freq_features
    kmer_feat_arr = kmer_features
    true_arr = true_labels
    
    filtered_arrs = filter_nctc_features(freq_features, kmer_features, 
                                         true_labels)
    freq_feat_arr, kmer_feat_arr, true_arr = filtered_arrs
    
    #labeled_true_arrs = large_indels_true_nctc_features(freq_features, 
    #                                                    kmer_features, 
    #                                                    true_labels)
    #freq_feat_arr, kmer_feat_arr, true_arr = labeled_true_arrs
    return freq_feat_arr, kmer_feat_arr, true_arr

def filter_nctc_features(freq_features, kmer_features, true_labels):
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
    true_arr = np.concatenate((true_labels[2000:19610], 
                               true_labels[21050:21865], 
                               true_labels[23975:31860], 
                               true_labels[33185:]))
    return freq_feat_arr, kmer_feat_arr, true_arr

def large_indels_true_nctc_features(freq_features, kmer_features, true_labels):
    large_indels = [(0, 2000), (19610, 21050), (21865, 23975), (31860, 33185)]
    non_indel_inds = [(2000, 19610), (21050, 21865), (23975, 31860), (33185, )]
    freq_feat_arr = freq_features
    kmer_feat_arr = kmer_features
    true_arr = np.concatenate((np.ones(2000),
                               true_labels[2000:19610], 
                               np.ones(21050 - 19610),
                               true_labels[21050:21865], 
                               np.ones(23975 - 21865),
                               true_labels[23975:31860], 
                               np.ones(33185 - 31860),
                               true_labels[33185:]))
    return freq_feat_arr, kmer_feat_arr, true_arr

def sim_a_20k_x_15_train(k, freq_width, kmer_width):
    int_pos_folder = os.path.join("..", "kmer_position_confirmation", 
                                  "simulation.integrated_pos.output")
    freq_file = os.path.join(int_pos_folder, 
                             "divergence_frequencies.txt")
    pk_out_file = os.path.join(int_pos_folder, 
                               "pos_to_kmer_output.txt")
    pos_out_file = os.path.join(int_pos_folder, 
                                "all_pos_output.txt")
    freq_out = utils.read_div_freq_file(freq_file)
    pk_out = read_pk_output(pk_out_file)
    pos_out = read_pos_output(pos_out_file)
    
    freq_features, kmer_features = get_features(freq_out, pk_out, k, 
                                                freq_width, kmer_width)
    true_labels = get_true_labels(pos_out)
    
    return freq_features, kmer_features, true_labels
    
def get_features(freq_out, pk_out, k, freq_width, kmer_width):
    pk_pos, pk_div, pk_counts, pk_kmers = pk_out
    freq_features = np.zeros(shape = (len(freq_out), freq_width))
    kmer_features = np.zeros(shape = (len(freq_out), kmer_width))
    for i, freq in enumerate(freq_out):
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
    return freq_features, kmer_features

def get_true_labels(pos_out):
    true_labels = np.array([-1 for _ in pos_out])
    for i, out in enumerate(pos_out):
        true = out[5].strip()
        if true == "YES":
            true_labels[i] = 1
        else:
            true_labels[i] = 0
    return true_labels

def make_predictions(clf, x, y):
    predictions = clf.predict(x)
    probs = clf.predict_proba(x)
    probs_first = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probs_first)
    roc_auc = auc(fpr, tpr)
    conf_mat = confusion_matrix(y, predictions)
    return predictions, probs, roc_auc, conf_mat
    
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