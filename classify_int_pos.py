# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:02:31 2018

@author: jeffrey_yuan
"""

import time, argparse, utils
import numpy as np
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
    parser.add_argument('-c', required=True, dest='classifier_file', 
                        default="",
                        metavar='classifier_file_input', action='store')
    parser.add_argument('-f', required=True, dest='freq_file', 
                        metavar='divergent_freq_file_input', action='store')
    parser.add_argument('-k', required=False, dest='pk_out_file', 
                        default="",
                        metavar='pos_to_kmer_output_file_input', action='store')
    parser.add_argument('-p', required=False, dest='pos_out_file', 
                        default="",
                        metavar='all_pos_output_file_input', action='store')
    args = parser.parse_args()
    
    #cov_out = read_cov_output(args.comp_cov_out_file)
    freq_outs = utils.read_div_freq_file(args.freq_file)
    pk_out = read_pk_output(args.pk_out_file)
    pk_pos, pk_div, pk_counts, pk_kmers = pk_out
    pos_out = read_pos_output(args.pos_out_file)
    #print pk_pos[:3]
    #print pk_div[:3]
    #print pk_counts[:3]
    #print pk_kmers[:3]
    
    k = 10
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
        
        #cov_pos, blasr_cov, freq_cov, sam_cov = cov_out[i]
        pkp = pk_pos[i]
        pkd = pk_div[i]
        pkc = pk_counts[i][:2 * k]
        pkk = pk_kmers[i][:2 * k]
        norm_pkc = [float(c) / cov for c in pkc]
        #if pos != cov_pos or pos != pkp:
        #    print "i", i, "pos", pos, "pkp", pkp
        #    print "cov:", "blasr", blasr_cov, "freq", freq_cov, "sam", sam_cov
        
        freq_fts = [cov, sub_ct, del_ct, ins_ct]
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
    print "all freq features", freq_features.shape
    print "all kmer features", kmer_features.shape
    print "all true labels", true_labels.shape
    print "all positions", all_pos.T.shape
    print "all tentative labels", tent_labels.T.shape
    
    
    num_pca_components = 5
    pca = PCA(n_components = num_pca_components)
    pca.fit(kmer_features)
    print "PCA explained variance", pca.explained_variance_ratio_
    print "PCA singular values", pca.singular_values_
    
    kmer_pca_features = pca.transform(kmer_features)
    
    all_features = np.concatenate((freq_features, kmer_pca_features), axis = 1)
    print "concatenated features", all_features.shape
    print all_features
    
    df = pandas.DataFrame(all_features)
    corr_mat = df.corr()
    #print corr_mat
    print "Features with corr > 0.5:"
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat[i])):
            if i != j and abs(corr_mat[i][j]) > 0.5:
                print labels[i], labels[j], corr_mat[i][j]
    print
    
    scaled_features = scale(all_features)
    print "scaled features", scaled_features.shape
    print scaled_features
    for i, f in enumerate(scaled_features[0]):
        print labels[i], sum(scaled_features[:, i])
    print
    
    #x_train = normalize(x_train_pca_concat, norm = "l1", axis = 0)
    #x_test = normalize(x_test_pca_concat, norm = "l1", axis = 0)
    #print "x_train", x_train
    #print "x_test", x_test
    #for i, f in enumerate(x_train[0]):
    #    print labels[i], sum(x_train[:, i])
    #print
    
    clf = joblib.load(args.classifier_file)
    print "Coefficients"
    for l, c in zip(labels, clf.coef_[0]):
        print "{0}\t{1}".format(l, c)
    print
    
    predictions = clf.predict(scaled_features)
    #print "Preds", predictions
    probs = clf.predict_proba(scaled_features)
    #print "Probs", probs
    probs_first = probs[:, 1]
    print "Score", clf.score(scaled_features, true_labels)
    fpr, tpr, thresholds = roc_curve(true_labels, probs_first)
    roc_auc = auc(fpr, tpr)
    conf_mat = confusion_matrix(true_labels, predictions)
    print "Confusion Matrix"
    print "TN FP"
    print "FN TP"
    print conf_mat
    print
    
    print "Pos", all_pos
    print "Tentative labels", tent_labels
    tent_conf_mat = confusion_matrix(true_labels, tent_labels)
    print "Tentative Confusion Matrix"
    print "TN FP"
    print "FN TP"
    print tent_conf_mat
    print
    
    plt.figure(1)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
    pfn = []
    pfp = []
    tfn = []
    tfp = []
    for pos, pred, true, tent in zip(all_pos, predictions, true_labels, tent_labels):
        if pred == 1 and true == 0:
            pfp.append(1)
        else:
            pfp.append(0)
        if pred == 0 and true == 1:
            pfn.append(2)
        else:
            pfn.append(0)
        if tent == 1 and true == 0:
            tfp.append(3)
        else:
            tfp.append(0)
        if tent == 0 and true == 1:
            tfn.append(4)
        else:
            tfn.append(0)

    plt.figure(2)
    plt.title('False Calls')
    plt.plot(all_pos, pfn, '.g', label = 'Prediction False Negatives')
    plt.plot(all_pos, pfp, '.b', label = 'Prediction False Positives')
    plt.plot(all_pos, tfn, '.r', label = 'Tentative False Negatives')
    plt.plot(all_pos, tfp, '.m', label = 'Tentative False Positives')
    plt.ylim(0, 6)
    plt.legend()
    plt.show()

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

def plot_and_write_low_boths(k, kmer_out, pos_out):
    fignum = 12
    fs = (20, 5)
    low_end = 50
    low_both_file = os.path.join("..", "kmer_position_confirmation", 
                              "nctc9964.repeat.5.forward.integrated_pos.with.polished.cons.2.as.true.output", 
                              "low_both_output.txt")
    low_both_occ_kmers = {}
    for out in kmer_out:
        kmer = out[0].strip()
        pos = out[1]
        count = out[2]
        div = out[3].strip()
        true = out[6].strip()
        occ = out[7].strip()
        if occ == "Both" and count < low_end:
            low_both_occ_kmers[kmer] = (pos, count, div, true)
    
    pos_vals = {}
    with open(low_both_file, "w") as f:
        f.write("{0:{km}}\t{1:5}\t{2:5}\tDiv\tTrue\tOcc\n".format("Kmer", "Pos", 
                                                                "Count", km = k))
        for kmer in sorted(low_both_occ_kmers, 
                           key = lambda x:low_both_occ_kmers[x][1], 
                           reverse = True):
            pos, count, div, true = low_both_occ_kmers[kmer]
            occ = "Both"
            f.write("{0:{km}}\t{1:5}\t{2:5}\t{3}\t{4}\t{5}\n".format(
                kmer, pos, count, div, true, occ, km = k))
            rounded_pos = int(round(pos))
            pos_vals[rounded_pos] = (kmer, count, div, true, occ)
    
    all_pos = [[] for _ in range(5)]
    med_counts = [[] for _ in range(5)]
    min_counts = [[] for _ in range(5)]
    max_counts = [[] for _ in range(5)]
    kmer_counts = [[] for _ in range(5)]
    for out in pos_out:
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        med = out[7]
        mn = out[8]
        mx = out[9]
        if pos in pos_vals:
            kmer, count, div, true, occ = pos_vals[pos]
            ind = count / 10
            all_pos[ind].append(pos)
            med_counts[ind].append(med)
            min_counts[ind].append(mn)
            max_counts[ind].append(mx)
            kmer_counts[ind].append(count)
    
    xlab = "Position"
    ylab = "{km}-mer counts".format(km = k)
    count_leg = ["0 - 9", "10 - 19", "20 - 29", "30 - 39", "40 - 49"]

    pylab.figure(fignum, figsize = fs)
    for i, c in enumerate(["g", "brown", "deepskyblue", 
                           "blue", "mediumpurple"]):
        #pylab.plot(all_pos[i], med_counts[i], ".", color = c)
        pylab.plot(all_pos[i], kmer_counts[i], ".", color = c)
    pylab.legend(count_leg)
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.title("Low, nondivergent kmer counts by position")
    
    pylab.figure(fignum + 1, figsize = fs)
    for i, c in enumerate(["g", "brown", "deepskyblue", 
                           "blue", "mediumpurple"]):
        pylab.plot(all_pos[i], med_counts[i], ".", color = c)
        #pylab.plot(all_pos[i], kmer_counts[i], ".", color = c)
    pylab.legend(count_leg)
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.title("Position scores of low, nondivergent kmers")
    

def plot_kmers(gen_out_file, k, kmer_out):
    split = 0
    gen_fig = 1    
    tent_fig = 2
    true_fig = 3
    occ_fig = 4
    gen_leg = ["general", "gen binned"]
    tent_leg = ["tentative div", "tent div binned", 
                "tentative nondiv", "tent nondiv binned"]
    true_leg = ["true div", "true div binned", 
                "true nondiv", "true nondiv binned"]
    occ_leg = ["No occurrences", "No occ binned", 
               "Occ in copy 1", "Occ 1 binned", 
               "Occ in copy 2", "Occ 2 binned", 
               "Occ in both", "Both occ binned"]
    total_legend = gen_leg + tent_leg + true_leg + occ_leg
    xlab = "{km}-mer frequency".format(km = k)
    ylab = "{km}-mer counts".format(km = k)
    #simulation
    sim_all_lims = {"ymin" : 0, 
                "ymax" : 1500, 
                "xmin" : 10, 
                "xmax" : 110}
    sim_div_lims = {"ymin" : 0, 
                "ymax" : 60, 
                "xmin" : 10, 
                "xmax" : 35}
    #NCTC9964
    nctc_all_lims = {"ymin" : 0,
                 "ymax" : 1200,
                 "xmin" : 15,
                 "xmax" : 110}
    lims = nctc_all_lims
    """
    plot_gen_output(gen_out_file, k, split, gen_fig)
    pylab.legend(gen_leg, 
                 fontsize = 12)
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.xlim(lims["xmin"], lims["xmax"])
    pylab.ylim(lims["ymin"], lims["ymax"])
    
    plot_tentative_kmers(kmer_out, split, tent_fig)
    pylab.legend(tent_leg, 
                 fontsize = 12)
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.xlim(lims["xmin"], lims["xmax"])
    pylab.ylim(lims["ymin"], lims["ymax"])
    
    plot_true_kmers(kmer_out, split, true_fig)
    pylab.legend(true_leg, 
                 fontsize = 12)
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.xlim(lims["xmin"], lims["xmax"])
    pylab.ylim(lims["ymin"], lims["ymax"])
    """
    plot_occ_kmers(kmer_out, split, occ_fig)
    pylab.legend(occ_leg, 
                 fontsize = 12)
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.xlim(lims["xmin"], lims["xmax"])
    pylab.ylim(lims["ymin"], lims["ymax"])
    #pylab.legend(total_legend, 
    #             fontsize = 12)
    
def plot_pos(pos_out, k, cov_out):
    all_fig_med = 5
    all_fig_max = 6
    tent_fig_med = 7
    tent_fig_max = 8
    cov_fig = 9
    true_fig_med = 10
    true_fig_max = 11
    all_med_leg = ["all median"]
    all_max_leg = ["all max"]
    tent_med_leg = ["tentative nondiv median", "tentative near div median", 
                    "tentative div median"]
    tent_max_leg = ["tentative nondiv max", "tentative near div max", 
                    "tentative div max"]
    true_med_leg = ["true nondiv median", "true near div median", 
                    "true div median"]
    true_max_leg = ["true nondiv max", "true near div max", 
                    "true div max"]
    cov_leg = ["coverage"]
    xlab = "Position"
    ylab = "{km}-mer counts".format(km = k)
    #simulation
    sim_lims = {"ymin" : 0, 
                "ymax" : 80}
    #NCTC9964
    nctc_lims = {"ymin" : 0,
                 "ymax" : 125}
    lims = nctc_lims
    #Default
    #fs = None
    fs = (20, 5)
    """
    plot_all_pos_med(pos_out, all_fig_med, fs)
    pylab.legend(all_med_leg, fontsize = 12, loc = "lower right")
    pylab.title("All Positions, Median {km}-mer Counts".format(km = k))
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.ylim(lims["ymin"], lims["ymax"])
    
    plot_all_pos_max(pos_out, all_fig_max, fs)
    pylab.legend(all_max_leg, fontsize = 12, loc = "lower right")
    pylab.title("All Positions, Max {km}-mer Counts".format(km = k))
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.ylim(lims["ymin"], lims["ymax"])
    """
    plot_tentative_pos_med(pos_out, tent_fig_med, fs)
    pylab.legend(tent_med_leg, fontsize = 12, loc = "lower right")
    pylab.title("Tentative Div vs Near-Div vs Non-Div Positions, Median Counts")
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.ylim(lims["ymin"], lims["ymax"])
    """
    plot_tentative_pos_max(pos_out, tent_fig_max, fs)
    pylab.legend(tent_max_leg, fontsize = 12, loc = "lower right")
    pylab.title("Tentative Div vs Near-Div vs Non-Div Positions, Max Counts")
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.ylim(lims["ymin"], lims["ymax"])
    """
    plot_coverage(cov_out, cov_fig, fs)
    pylab.legend(cov_leg, fontsize = 12, loc = "lower right")
    pylab.title("Coverage by Position")
    pylab.xlabel(xlab)
    pylab.ylabel("Coverage")
    
    plot_true_pos_med(pos_out, true_fig_med, fs)
    pylab.legend(true_med_leg, fontsize = 12, loc = "lower right")
    pylab.title("True Div vs Near-Div vs Non-Div Positions, Median Counts")
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.ylim(lims["ymin"], lims["ymax"])
    """
    plot_true_pos_max(pos_out, true_fig_max, fs)
    pylab.legend(true_max_leg, fontsize = 12, loc = "lower right")
    pylab.title("True Div vs Near-Div vs Non-Div Positions, Max Counts")
    pylab.xlabel(xlab)
    pylab.ylabel(ylab)
    pylab.ylim(lims["ymin"], lims["ymax"])
    """

def plot_all_pos_med(pos_out, fignum_med, fs):
    all_pos = []
    med_counts = []
    min_counts = []
    max_counts = []
    for out in pos_out:
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        med = out[7]
        mn = out[8]
        mx = out[9]
        all_pos.append(pos)
        med_counts.append(med)
        min_counts.append(mn)
        max_counts.append(mx)
    
    pylab.figure(fignum_med, figsize = fs)
    pylab.plot(all_pos, med_counts, ".b")
    #pylab.plot(all_pos, min_counts, "vc")
    #pylab.figure(fignum_max)
    #pylab.plot(all_pos, max_counts, ".c")

def plot_all_pos_max(pos_out, fignum_max, fs):
    all_pos = []
    med_counts = []
    min_counts = []
    max_counts = []
    for out in pos_out:
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        med = out[7]
        mn = out[8]
        mx = out[9]
        all_pos.append(pos)
        med_counts.append(med)
        min_counts.append(mn)
        max_counts.append(mx)
    
    #pylab.figure(fignum_med)
    #pylab.plot(all_pos, med_counts, ".b")
    #pylab.plot(all_pos, min_counts, "vc")
    pylab.figure(fignum_max, figsize = fs)
    pylab.plot(all_pos, max_counts, ".c")

def plot_tentative_pos_med(pos_out, fignum_med, fs):
    tent_div_pos = []
    tent_near_div_pos = []
    tent_nondiv_pos = []
    tent_div_med_counts = []
    tent_div_min_counts = []
    tent_div_max_counts = []
    tent_near_div_med_counts = []
    tent_near_div_min_counts = []
    tent_near_div_max_counts = []
    tent_nondiv_med_counts = []
    tent_nondiv_min_counts = []
    tent_nondiv_max_counts = []
    for out in pos_out:
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        med = out[7]
        mn = out[8]
        mx = out[9]
        if div == "YES":
            tent_div_pos.append(pos)
            tent_div_med_counts.append(med)
            tent_div_min_counts.append(mn)
            tent_div_max_counts.append(mx)
        elif near_div == "YES":
            tent_near_div_pos.append(pos)
            tent_near_div_med_counts.append(med)
            tent_near_div_min_counts.append(mn)
            tent_near_div_max_counts.append(mx)
        else:
            tent_nondiv_pos.append(pos)
            tent_nondiv_med_counts.append(med)
            tent_nondiv_min_counts.append(mn)
            tent_nondiv_max_counts.append(mx)
    """nondiv_sample = random.sample(range(len(tent_nondiv_pos)), 
                                  2 * len(tent_near_div_pos))
    sampled_non_pos = [tent_nondiv_pos[p] for p in nondiv_sample]
    sampled_non_med_counts = [tent_nondiv_med_counts[p] for p in nondiv_sample]
    sampled_non_min_counts = [tent_nondiv_min_counts[p] for p in nondiv_sample]
    sampled_non_max_counts = [tent_nondiv_max_counts[p] for p in nondiv_sample]"""
    
    pylab.figure(fignum_med, figsize = fs)
    pylab.plot(tent_nondiv_pos, tent_nondiv_med_counts, ".g")
    #pylab.plot(sampled_non_pos, sampled_non_med_counts, ".b")
    pylab.plot(tent_near_div_pos, tent_near_div_med_counts, ".b")
    pylab.plot(tent_div_pos, tent_div_med_counts, "or")
    
    #pylab.figure(fignum_max)
    #pylab.plot(tent_nondiv_pos, tent_nondiv_max_counts, ",y")
    #pylab.plot(sampled_non_pos, sampled_non_min_counts, ".r")
    #pylab.plot(tent_near_div_pos, tent_near_div_max_counts, ".c")
    #pylab.plot(tent_div_pos, tent_div_max_counts, "om")
    
    #pylab.plot(tent_nondiv_pos, tent_nondiv_min_counts, "vg")
    #pylab.plot(sampled_non_pos, sampled_non_max_counts, ".g")
    #pylab.plot(tent_near_div_pos, tent_near_div_min_counts, "vc")
    #pylab.plot(tent_div_pos, tent_div_min_counts, "vm")

def plot_tentative_pos_max(pos_out, fignum_max, fs):
    tent_div_pos = []
    tent_near_div_pos = []
    tent_nondiv_pos = []
    tent_div_med_counts = []
    tent_div_min_counts = []
    tent_div_max_counts = []
    tent_near_div_med_counts = []
    tent_near_div_min_counts = []
    tent_near_div_max_counts = []
    tent_nondiv_med_counts = []
    tent_nondiv_min_counts = []
    tent_nondiv_max_counts = []
    for out in pos_out:
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        med = out[7]
        mn = out[8]
        mx = out[9]
        if div == "YES":
            tent_div_pos.append(pos)
            tent_div_med_counts.append(med)
            tent_div_min_counts.append(mn)
            tent_div_max_counts.append(mx)
        elif near_div == "YES":
            tent_near_div_pos.append(pos)
            tent_near_div_med_counts.append(med)
            tent_near_div_min_counts.append(mn)
            tent_near_div_max_counts.append(mx)
        else:
            tent_nondiv_pos.append(pos)
            tent_nondiv_med_counts.append(med)
            tent_nondiv_min_counts.append(mn)
            tent_nondiv_max_counts.append(mx)
    """nondiv_sample = random.sample(range(len(tent_nondiv_pos)), 
                                  2 * len(tent_near_div_pos))
    sampled_non_pos = [tent_nondiv_pos[p] for p in nondiv_sample]
    sampled_non_med_counts = [tent_nondiv_med_counts[p] for p in nondiv_sample]
    sampled_non_min_counts = [tent_nondiv_min_counts[p] for p in nondiv_sample]
    sampled_non_max_counts = [tent_nondiv_max_counts[p] for p in nondiv_sample]"""
    
    #pylab.figure(fignum_med)
    #pylab.plot(tent_nondiv_pos, tent_nondiv_med_counts, ",g")
    #pylab.plot(sampled_non_pos, sampled_non_med_counts, ".b")
    #pylab.plot(tent_near_div_pos, tent_near_div_med_counts, ".b")
    #pylab.plot(tent_div_pos, tent_div_med_counts, "or")
    
    pylab.figure(fignum_max, figsize = fs)
    pylab.plot(tent_nondiv_pos, tent_nondiv_max_counts, ".y")
    #pylab.plot(sampled_non_pos, sampled_non_min_counts, ".r")
    pylab.plot(tent_near_div_pos, tent_near_div_max_counts, ".c")
    pylab.plot(tent_div_pos, tent_div_max_counts, "om")
    
    #pylab.plot(tent_nondiv_pos, tent_nondiv_min_counts, "vg")
    #pylab.plot(sampled_non_pos, sampled_non_max_counts, ".g")
    #pylab.plot(tent_near_div_pos, tent_near_div_min_counts, "vc")
    #pylab.plot(tent_div_pos, tent_div_min_counts, "vm")

def plot_coverage(cov_out, fignum, fs):
    pos, cov = zip(*cov_out)
    pylab.figure(fignum, figsize = fs)
    pylab.plot(pos, cov, ".k")

def plot_true_pos_med(pos_out, fignum_med, fs):
    true_div_pos = []
    true_near_div_pos = []
    true_nondiv_pos = []
    true_div_med_counts = []
    true_div_min_counts = []
    true_div_max_counts = []
    true_near_div_med_counts = []
    true_near_div_min_counts = []
    true_near_div_max_counts = []
    true_nondiv_med_counts = []
    true_nondiv_min_counts = []
    true_nondiv_max_counts = []
    for out in pos_out:
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        med = out[7]
        mn = out[8]
        mx = out[9]
        if true == "YES":
            true_div_pos.append(pos)
            true_div_med_counts.append(med)
            true_div_min_counts.append(mn)
            true_div_max_counts.append(mx)
        elif near_true == "YES":
            true_near_div_pos.append(pos)
            true_near_div_med_counts.append(med)
            true_near_div_min_counts.append(mn)
            true_near_div_max_counts.append(mx)
        else:
            true_nondiv_pos.append(pos)
            true_nondiv_med_counts.append(med)
            true_nondiv_min_counts.append(mn)
            true_nondiv_max_counts.append(mx)
    """nondiv_sample = random.sample(range(len(true_nondiv_pos)), 
                                  2 * len(true_near_div_pos))
    sampled_non_pos = [true_nondiv_pos[p] for p in nondiv_sample]
    sampled_non_med_counts = [true_nondiv_med_counts[p] for p in nondiv_sample]
    sampled_non_min_counts = [true_nondiv_min_counts[p] for p in nondiv_sample]
    sampled_non_max_counts = [true_nondiv_max_counts[p] for p in nondiv_sample]"""
    
    pylab.figure(fignum_med, figsize = fs)
    pylab.plot(true_nondiv_pos, true_nondiv_med_counts, ".g")
    #pylab.plot(sampled_non_pos, sampled_non_med_counts, ".b")
    pylab.plot(true_near_div_pos, true_near_div_med_counts, ".b")
    pylab.plot(true_div_pos, true_div_med_counts, "or")
    
    #pylab.figure(fignum_max)
    #pylab.plot(true_nondiv_pos, true_nondiv_max_counts, ",y")
    #pylab.plot(sampled_non_pos, sampled_non_min_counts, ".r")
    #pylab.plot(true_near_div_pos, true_near_div_max_counts, ".c")
    #pylab.plot(true_div_pos, true_div_max_counts, "om")
    
    #pylab.plot(true_nondiv_pos, true_nondiv_min_counts, "vg")
    #pylab.plot(sampled_non_pos, sampled_non_max_counts, ".g")
    #pylab.plot(true_near_div_pos, true_near_div_min_counts, "vc")
    #pylab.plot(true_div_pos, true_div_min_counts, "vm")

def plot_true_pos_max(pos_out, fignum_max, fs):
    true_div_pos = []
    true_near_div_pos = []
    true_nondiv_pos = []
    true_div_med_counts = []
    true_div_min_counts = []
    true_div_max_counts = []
    true_near_div_med_counts = []
    true_near_div_min_counts = []
    true_near_div_max_counts = []
    true_nondiv_med_counts = []
    true_nondiv_min_counts = []
    true_nondiv_max_counts = []
    for out in pos_out:
        pos = out[0]
        div = out[1].strip()
        near_div = out[2].strip()
        true = out[5].strip()
        near_true = out[6].strip()
        med = out[7]
        mn = out[8]
        mx = out[9]
        if true == "YES":
            true_div_pos.append(pos)
            true_div_med_counts.append(med)
            true_div_min_counts.append(mn)
            true_div_max_counts.append(mx)
        elif near_true == "YES":
            true_near_div_pos.append(pos)
            true_near_div_med_counts.append(med)
            true_near_div_min_counts.append(mn)
            true_near_div_max_counts.append(mx)
        else:
            true_nondiv_pos.append(pos)
            true_nondiv_med_counts.append(med)
            true_nondiv_min_counts.append(mn)
            true_nondiv_max_counts.append(mx)
    """nondiv_sample = random.sample(range(len(true_nondiv_pos)), 
                                  2 * len(true_near_div_pos))
    sampled_non_pos = [true_nondiv_pos[p] for p in nondiv_sample]
    sampled_non_med_counts = [true_nondiv_med_counts[p] for p in nondiv_sample]
    sampled_non_min_counts = [true_nondiv_min_counts[p] for p in nondiv_sample]
    sampled_non_max_counts = [true_nondiv_max_counts[p] for p in nondiv_sample]"""
    
    #pylab.figure(fignum_med)
    #pylab.plot(true_nondiv_pos, true_nondiv_med_counts, ",g")
    #pylab.plot(sampled_non_pos, sampled_non_med_counts, ".b")
    #pylab.plot(true_near_div_pos, true_near_div_med_counts, ".b")
    #pylab.plot(true_div_pos, true_div_med_counts, "or")
    
    pylab.figure(fignum_max, figsize = fs)
    pylab.plot(true_nondiv_pos, true_nondiv_max_counts, ".y")
    #pylab.plot(sampled_non_pos, sampled_non_min_counts, ".r")
    pylab.plot(true_near_div_pos, true_near_div_max_counts, ".c")
    pylab.plot(true_div_pos, true_div_max_counts, "om")
    
    #pylab.plot(true_nondiv_pos, true_nondiv_min_counts, "vg")
    #pylab.plot(sampled_non_pos, sampled_non_max_counts, ".g")
    #pylab.plot(true_near_div_pos, true_near_div_min_counts, "vc")
    #pylab.plot(true_div_pos, true_div_min_counts, "vm")
    

def plot_gen_output(gen_file, k, split, fignum):
    gen_stats, count_nums = read_gen_output(gen_file)
    print "General stats output:"    
    print "Repeat Length\t", gen_stats["rep_len"]
    print "Reads N50\t\t", gen_stats["n50"]
    print "Average Coverage\t", gen_stats["av_cov"]
    print "k\t\t", k
    print "Split\t\t", split
    
    counts = get_counts(count_nums, split)
    low_counts, low_nums, high_counts, high_nums = counts
    
    bin_size = 4
    high_inds, high_bins = bin_counts(high_counts, high_nums, bin_size)
    
    pylab.figure(fignum)
    pylab.plot(high_counts, [h for h in high_nums], "c.")
    pylab.plot(high_inds, [h for h in high_bins], "-b")

def plot_tentative_kmers(kmer_out, split, fignum):
    tent_div_counts = {}
    tent_nondiv_counts = {}
    for out in kmer_out:
        kmer = out[0].strip()
        count = out[2]
        div = out[3].strip()
        true = out[6].strip()
        occ = out[7].strip()
        if div == "YES":
            if count not in tent_div_counts:
                tent_div_counts[count] = 1
            else:
                tent_div_counts[count] += 1
        else:
            if count not in tent_nondiv_counts:
                tent_nondiv_counts[count] = 1
            else:
                tent_nondiv_counts[count] += 1
                
    td_counts = get_counts(tent_div_counts, split)
    low_td_counts, low_td_nums, high_td_counts, high_td_nums = td_counts
    tnd_counts = get_counts(tent_nondiv_counts, split)
    low_tnd_counts, low_tnd_nums, high_tnd_counts, high_tnd_nums = tnd_counts
    
    bin_size = 4
    high_td_inds, high_td_bins = bin_counts(high_td_counts, high_td_nums, bin_size)
    high_tnd_inds, high_tnd_bins = bin_counts(high_tnd_counts, high_tnd_nums, bin_size)
    
    pylab.figure(fignum)
    pylab.plot(high_td_counts, [h for h in high_td_nums], "m.")
    pylab.plot(high_td_inds, [h for h in high_td_bins], "-r")
    pylab.plot(high_tnd_counts, [h for h in high_tnd_nums], "c.")
    pylab.plot(high_tnd_inds, [h for h in high_tnd_bins], "-b")

def plot_true_kmers(kmer_out, split, fignum):
    true_div_counts = {}
    true_nondiv_counts = {}
    for out in kmer_out:
        kmer = out[0].strip()
        count = out[2]
        div = out[3].strip()
        true = out[6].strip()
        occ = out[7].strip()
        if true == "YES":
            if count not in true_div_counts:
                true_div_counts[count] = 1
            else:
                true_div_counts[count] += 1
        else:
            if count not in true_nondiv_counts:
                true_nondiv_counts[count] = 1
            else:
                true_nondiv_counts[count] += 1
                
    td_counts = get_counts(true_div_counts, split)
    low_td_counts, low_td_nums, high_td_counts, high_td_nums = td_counts
    tnd_counts = get_counts(true_nondiv_counts, split)
    low_tnd_counts, low_tnd_nums, high_tnd_counts, high_tnd_nums = tnd_counts
    
    bin_size = 4
    high_td_inds, high_td_bins = bin_counts(high_td_counts, high_td_nums, bin_size)
    high_tnd_inds, high_tnd_bins = bin_counts(high_tnd_counts, high_tnd_nums, bin_size)
    
    pylab.figure(fignum)
    pylab.plot(high_td_counts, [h for h in high_td_nums], "m.")
    pylab.plot(high_td_inds, [h for h in high_td_bins], "-r")
    pylab.plot(high_tnd_counts, [h for h in high_tnd_nums], "c.")
    pylab.plot(high_tnd_inds, [h for h in high_tnd_bins], "-b")

def plot_occ_kmers(kmer_out, split, fignum):
    occ_none_counts = {}
    occ_1_counts = {}
    occ_2_counts = {}
    occ_both_counts = {}
    for out in kmer_out:
        kmer = out[0].strip()
        count = out[2]
        div = out[3].strip()
        true = out[6].strip()
        occ = out[7].strip()
        if occ == "None":
            if count not in occ_none_counts:
                occ_none_counts[count] = 1
            else:
                occ_none_counts[count] += 1
        elif occ == "1":
            if count not in occ_1_counts:
                occ_1_counts[count] = 1
            else:
                occ_1_counts[count] += 1
        elif occ == "2":
            if count not in occ_2_counts:
                occ_2_counts[count] = 1
            else:
                occ_2_counts[count] += 1
        elif occ == "Both":
            if count not in occ_both_counts:
                occ_both_counts[count] = 1
            else:
                occ_both_counts[count] += 1
                
    on_counts = get_counts(occ_none_counts, split)
    low_on_counts, low_on_nums, high_on_counts, high_on_nums = on_counts
    o1_counts = get_counts(occ_1_counts, split)
    low_o1_counts, low_o1_nums, high_o1_counts, high_o1_nums = o1_counts
    o2_counts = get_counts(occ_2_counts, split)
    low_o2_counts, low_o2_nums, high_o2_counts, high_o2_nums = o2_counts
    ob_counts = get_counts(occ_both_counts, split)
    low_ob_counts, low_ob_nums, high_ob_counts, high_ob_nums = ob_counts
    
    bin_size = 4
    high_on_inds, high_on_bins = bin_counts(high_on_counts, high_on_nums, bin_size)
    high_o1_inds, high_o1_bins = bin_counts(high_o1_counts, high_o1_nums, bin_size)
    high_o2_inds, high_o2_bins = bin_counts(high_o2_counts, high_o2_nums, bin_size)
    high_ob_inds, high_ob_bins = bin_counts(high_ob_counts, high_ob_nums, bin_size)
    
    pylab.figure(fignum)
    pylab.plot(high_on_counts, [h for h in high_on_nums], "r.")
    pylab.plot(high_on_inds, [h for h in high_on_bins], "-r")
    pylab.plot(high_o1_counts, [h for h in high_o1_nums], "b.")
    pylab.plot(high_o1_inds, [h for h in high_o1_bins], "-b")
    pylab.plot(high_o2_counts, [h for h in high_o2_nums], "g.")
    pylab.plot(high_o2_inds, [h for h in high_o2_bins], "-g")
    pylab.plot(high_ob_counts, [h for h in high_ob_nums], "k.")
    pylab.plot(high_ob_inds, [h for h in high_ob_bins], "-k")

def read_gen_output(out_file):
    gen_stats = {}
    count_nums = {}
    with open(out_file, "r") as f:
        count_bool = False
        for i, line in enumerate(f):
            line = line.strip()
            parts = line.split("\t")
            if line and parts and parts[0]:
                if parts[0].strip() == "Rep Len":
                    gen_stats["rep_len"] = int(float(parts[1]))
                elif parts[0].strip() == "Reads N50":
                    gen_stats["n50"] = int(float(parts[1]))
                elif parts[0].strip() == "Av Cov":
                    gen_stats["av_cov"] = float(parts[1])
                elif parts[0].strip() == "Count":
                    count_bool = True
                elif count_bool:
                    count_nums[int(parts[0])] = int(parts[1])
    return gen_stats, count_nums

def read_kmer_output(kmer_file):
    kmer_out = []
    with open(kmer_file, "r") as f:
        int_inds = [1, 2]
        float_inds = [5]
        for i, line in enumerate(f):
            line = line.strip()
            if i > 0:
                parts = line.split("\t")
                for ind in int_inds:
                    parts[ind] = int(round(float(parts[ind])))
                for ind in float_inds:
                    parts[ind] = float(parts[ind])
                kmer_out.append(parts)
    return kmer_out

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
    

#OLDER FUNCTIONS
def del_unspaced_positions(positions):
    pos = sorted(positions)
    new_positions = []
    prev_unspaced = False
    for i in range(len(pos) - 1):
        curr_pos = pos[i]
        next_pos = pos[i + 1]
        if next_pos - curr_pos == 1:
            prev_unspaced = True
        else:
            if not prev_unspaced:
                new_positions.append(curr_pos)
            prev_unspaced = False
    if not prev_unspaced:
        new_positions.append(pos[-1])
    return new_positions

def get_counts(count_nums, split):
    low_counts = []
    low_nums = []
    high_counts = []
    high_nums = []
    for c in sorted(count_nums):
        if c < split:
            low_counts.append(c)
            low_nums.append(count_nums[c])
        elif c >= split:
            high_counts.append(c)
            high_nums.append(count_nums[c])
    
    return low_counts, low_nums, high_counts, high_nums

def bin_counts(counts, nums, bin_size):
    inds = []
    bins = []
    
    for i in range(len(counts) / bin_size + 1):
        ind = i * bin_size
        if ind < len(counts):
            inds.append(counts[ind])
            bins.append(np.mean(nums[ind:ind + bin_size]))
    
    return inds, bins

def binomial_model():
    k = 10
    leng = 200 - k + 1
    e = 0.12
    #p = (1-e)**k
    survival_rate = 0.3
    p = survival_rate
    cov = 140
    print p
    print cov
    
    x = scipy.linspace(0, 80, 81)
    
    err_pmf = scipy.stats.binom.pmf(x, cov, 0.25**k)*leng*cov
    uni_pmf = scipy.stats.binom.pmf(x, cov, p)
    
    sum_pmf = uni_pmf + err_pmf
    
    return x, err_pmf, uni_pmf, sum_pmf, leng

def mean_model():
    mean = 42.66
    std = 8.88
    x = scipy.linspace(0, 80, 81)
    norm_pdf = scipy.stats.norm.pdf(x, loc = mean, scale = std)
    return x, norm_pdf

def sample_from_pmf(x, pmf, num_samples):
    count_lst = np.random.choice(x, size = num_samples, p = pmf)
    counts = dict()
    for i in count_lst:
        c = int(i)
        counts[c] = counts.get(c, 0) + 1
    return counts

def print_counts(counts):
    print "Count\tNum kmers"
    for c in counts:
        print "{0}\t{1}".format(c, counts[c])
    
def plot_model(pmf):
    pylab.plot(pmf)
    
def plotPMFTrue(pmf,freqs,counts):
    pylab.close()
    pylab.plot(pmf,color='g',linestyle='--')
    pylab.plot(freqs,counts,color='g')
    #pylab.yscale('log')
    pylab.yscale('linear')
    pylab.ylim(1,1*10**7)
    pylab.xlim(0,50)

def print_pmf(x, pmf):
    print 'Surviving kmers\tProbability'
    for freq, num in zip(x, pmf):
        #val = int(num)
        val = num
        if round(val, 4) > 0.0:
            print '%d\t%0.4f' % (freq, val)
    print

if __name__ == "__main__":
    main()