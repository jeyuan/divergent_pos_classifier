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
from sklearn.preprocessing import normalize, scale
import matplotlib.pyplot as plt
import pandas
import pickle
from sklearn.externals import joblib

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', required=False, dest='comp_cov_out_file', 
                        default="",
                        metavar='comp_cov_output_file_input', action='store')
    parser.add_argument('-f', required=True, dest='freq_file', 
                        metavar='divergent_freq_file_input', action='store')
    parser.add_argument('-k', required=False, dest='pk_out_file', 
                        default="",
                        metavar='pos_to_kmer_output_file_input', action='store')
    parser.add_argument('-p', required=False, dest='pos_out_file', 
                        default="",
                        metavar='all_pos_output_file_input', action='store')
    args = parser.parse_args()
    
    k = 10
    large_indels = [(0, 2000), (19610, 21050), (21865, 23975), (31860, 33185)]
    #indel_inds = set()
    #for start, end in large_indels:
    #    indel_inds = indel_inds.union(set(range(start, end)))
    
    cov_out = read_cov_output(args.comp_cov_out_file)
    freq_outs = utils.read_div_freq_file(args.freq_file)
    pk_out = read_pk_output(args.pk_out_file)
    pk_pos, pk_div, pk_counts, pk_kmers = pk_out
    pos_out = read_pos_output(args.pos_out_file)
    #print pk_pos[:3]
    #print pk_div[:3]
    #print pk_counts[:3]
    #print pk_kmers[:3]
    
    freq_width = 4
    kmer_width = 2 * k
    freq_features = np.zeros(shape = (len(freq_outs), freq_width))
    kmer_features = np.zeros(shape = (len(freq_outs), kmer_width))
    for i, freq in enumerate(freq_outs):
        pos = freq[0]
        cov = freq[1]
        mat_base = freq[2]
        mat_ct = freq[3]
        sub_base = freq[4]
        sub_ct = freq[5]
        del_base = freq[6]
        del_ct = freq[7]
        ins_base = freq[8]
        ins_ct = freq[9]
        
        cov_pos, blasr_cov, freq_cov, sam_cov = cov_out[i]
        pkp = pk_pos[i]
        pkd = pk_div[i]
        pkc = pk_counts[i][:2 * k]
        pkk = pk_kmers[i][:2 * k]
        norm_pkc = [float(c) / cov for c in pkc]
        if pos != cov_pos or pos != pkp:
            print "i", i, "pos", pos, "pkp", pkp
            print "cov:", "blasr", blasr_cov, "freq", freq_cov, "sam", sam_cov
        
        freq_fts = [freq_cov, sub_ct, del_ct, ins_ct]
        freq_features[i][0:freq_width] = freq_fts
        kmer_features[i][0:len(norm_pkc)] = norm_pkc
    
    tent_labels = np.array([-1 for _ in pos_out])
    true_labels = np.array([-1 for _ in pos_out])
    all_pos = np.array([-1 for _ in pos_out])
    for i, out in enumerate(pos_out):
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        if true == "YES":
            true_labels[i] = 1
        else:
            true_labels[i] = 0
        all_pos[i] = pos
        if div == "YES":
            tent_labels[i] = 1
        else:
            tent_labels[i] = 0
    
    labels = ["cov", "sub", "del", "ins", 
              "kmer_pca1", "kmer_pca2", "kmer_pca3", "kmer_pca4", "kmer_pca5", 
              "kmer_pca6", "kmer_pca7", "kmer_pca8", "kmer_pca9", "kmer_pca10"]
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
    pos_arr = np.concatenate((all_pos[2000:19610], 
                              all_pos[21050:21865], 
                              all_pos[23975:31860], 
                              all_pos[33185:]))
    tent_arr = np.concatenate((tent_labels[2000:19610], 
                              tent_labels[21050:21865], 
                              tent_labels[23975:31860], 
                              tent_labels[33185:]))
    print "all freq features", freq_features.shape
    print "all kmer features", kmer_features.shape
    print "all true labels", true_labels.shape
    print "filtered freq features", freq_feat_arr.shape
    print "filtered kmer features", kmer_feat_arr.shape
    print "filtered true labels", lab_arr.shape
    print "filtered positions", pos_arr.T.shape
    print "filtered tentative labels", tent_arr.T.shape
    
    all_feat_arr = np.concatenate((freq_feat_arr, kmer_feat_arr), axis = 1)
    print "concatenated features", all_feat_arr.shape
    
    pos_2d = pos_arr[:, np.newaxis]
    tent_2d = tent_arr[:, np.newaxis]
    extra_feat_arr = np.concatenate((all_feat_arr, pos_2d, tent_2d), axis = 1)
    print "extra features", extra_feat_arr.shape
    print
    
    orig_x_train, orig_x_test, y_train, y_test = \
        train_test_split(extra_feat_arr, lab_arr, test_size = 0.25, 
                         random_state = 0)
    print "X", orig_x_train
    #print "y", y_train
    
    feat_tot = freq_width + kmer_width
    split_train = np.array_split(orig_x_train, [freq_width, feat_tot, feat_tot + 1], 
                                 axis = 1)
    x_freq_train = split_train[0]
    x_kmer_train = split_train[1]
    x_pos_train = split_train[2]
    x_tent_train = split_train[3]
    print "X freq", x_freq_train.shape
    print "X kmer", x_kmer_train.shape
    print
    split_test = np.array_split(orig_x_test, [freq_width, feat_tot, feat_tot + 1], 
                                 axis = 1)
    x_freq_test = split_test[0]
    x_kmer_test = split_test[1]
    x_pos_test = split_test[2]
    x_tent_test = split_test[3]
            
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
    df = pandas.DataFrame(x_train_pca_concat)
    corr_mat = df.corr()
    #print corr_mat
    print "Features with corr > 0.5:"
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat[i])):
            if i != j and abs(corr_mat[i][j]) > 0.5:
                print labels[i], labels[j], corr_mat[i][j]
    print
    
    x_train = scale(x_train_pca_concat)
    x_test = scale(x_test_pca_concat)
    print "x_train", x_train
    print "x_test", x_test
    for i, f in enumerate(x_train[0]):
        print labels[i], sum(x_train[:, i])
    print
    
    #x_train = normalize(x_train_pca_concat, norm = "l1", axis = 0)
    #x_test = normalize(x_test_pca_concat, norm = "l1", axis = 0)
    #print "x_train", x_train
    #print "x_test", x_test
    #for i, f in enumerate(x_train[0]):
    #    print labels[i], sum(x_train[:, i])
    #print
    
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
    #print "Train_preds", train_predictions
    train_probs = logRegr.predict_proba(x_train)
    #print "Train_probs", train_probs
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
    
    #print "Train pos", x_pos_train
    #print "Train tentative labels", x_tent_train
    train_tent_conf_mat = confusion_matrix(y_train, x_tent_train)
    print "Train tentative conf matrix"
    print "TN FP"
    print "FN TP"
    print train_tent_conf_mat
    print
    
    #Need to do cross validation on training set
    plt.figure(1)
    plt.title('ROC Curve for Training Set')
    plt.plot(train_fpr, train_tpr, 'b', label = 'AUC = %0.2f' % train_roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    test_predictions = logRegr.predict(x_test)
    #print "Test_preds", test_predictions
    test_probs = logRegr.predict_proba(x_test)
    #print "Test_probs", test_probs
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
    
    #print "Test pos", x_pos_test
    #print "Test tentative labels", x_tent_test
    test_tent_conf_mat = confusion_matrix(y_test, x_tent_test)
    print "Test tentative conf matrix"
    print "TN FP"
    print "FN TP"
    print test_tent_conf_mat
    print
    
    plt.figure(2)
    plt.title('ROC Curve for Test Set')
    plt.plot(test_fpr, test_tpr, 'b', label = 'AUC = %0.2f' % test_roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    int_pos_folder = "".join(["../kmer_position_confirmation/", 
                              "nctc9964.repeat.5.forward.integrated_pos", 
                              ".with.polished.cons.2.as.true.output/"])
    classifier_file = int_pos_folder + "nctc9964.rep_5.log_reg_classifier.sav"
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

def read_cov_output(cov_file):
    cov_out = []
    with open(cov_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i > 0:
                parts = map(int, line.split("\t"))
                cov_out.append(parts)
    return cov_out

if __name__ == "__main__":
    main()