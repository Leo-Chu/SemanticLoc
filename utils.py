
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def cdf_plot(true_result, est_result):

    fig, ax = plt.subplots(figsize=(8, 4))

    data1 = true_result-est_result
    error = np.sqrt(data1[:, 0]**2 + data1[:, 1]**2)

    count1, bins_count1 = np.histogram(error, bins=100)
    pdf1 = count1 / sum(count1)
    cdf1 = np.cumsum(pdf1)
    plt.plot(bins_count1[1:], cdf1, label="CDF1", color='blue', linewidth=2.7)
    plt.show()

    ax.set_title('Cumulative Distribution Function Curves')
    ax.set_xlabel('Localization Error(m)')
    ax.set_ylabel('Probability')

def cdfs_plot(true_result, est_result):

    fig, ax = plt.subplots(figsize=(8, 4))

    error = (true_result - est_result)**2
    countb1m1, bins_countb1m1 = np.histogram(np.squeeze(error[1, 0, :]), bins=100)
    countb1m2, bins_countb1m2 = np.histogram(np.squeeze(error[0, 1, :]), bins=100)
    countb2m1, bins_countb2m1 = np.histogram(np.squeeze(error[0, 0, :]), bins=100)
    countb2m2, bins_countb2m2 = np.histogram(np.squeeze(error[1, 1, :]), bins=100)

    pdfb1m1 = countb1m1 / sum(countb1m1)
    pdfb1m2 = countb1m2 / sum(countb1m2)
    pdfb2m1 = countb2m1 / sum(countb2m1)
    pdfb2m2 = countb2m2 / sum(countb2m2)

    cdfb1m1 = np.cumsum(pdfb1m1)
    cdfb1m2 = np.cumsum(pdfb1m2)
    cdfb2m1 = np.cumsum(pdfb2m1)
    cdfb2m2 = np.cumsum(pdfb2m2)

    plt.plot(bins_countb1m1[1:], cdfb1m1, label="DCNN", color='green', linestyle = 'dotted', linewidth=2.7)
    plt.plot(bins_countb1m2[1:], cdfb1m2, label="Proposed", color='black', linestyle = 'dashed', linewidth=2.7)
    plt.plot(bins_countb2m1[1:], cdfb2m1, label="DCNN(VP)", color='darkorange', linestyle = 'dashdot', linewidth=2.7)
    plt.plot(bins_countb2m2[1:], cdfb2m2, label="Proposed(VP)", color='purple', linestyle = 'solid', linewidth=2.7)
    plt.grid(True)
    plt.legend(loc=4)
    plt.show()
    plt.axis([0, 10, 0, 1])
    ax.set_title('Cumulative Distribution Function Curves')
    ax.set_xlabel('Localization Error(m)')
    ax.set_ylabel('Probability')


def result_plot(true_result, est_result):

    idx, scale = 30, 1
    xx1 = true_result[idx*100:(idx+1)*100, 0]*scale
    yy1 = true_result[idx*100:(idx+1)*100, 1]

    xx2 = est_result[idx*100:(idx+1)*100, 0]*scale
    yy2 = est_result[idx*100:(idx+1)*100, 1]

    plt.scatter(xx1, yy1, marker='o', alpha=0.65)
    plt.scatter(xx2, yy2, marker='+', alpha=0.65)


def PCP_merge(loc_PCP):

    LOS_loc, DNLOS_loc, SNLOS_loc = [], [], []
    m = loc_PCP.shape[0]

    for ii in range(m):
        if loc_PCP[ii, 2] == 0.0:
            LOS_loc.append(np.squeeze(loc_PCP[ii, :2]))
        elif loc_PCP[ii, 2] == 1.0:
            DNLOS_loc.append(loc_PCP[ii, :2])
        elif loc_PCP[ii, 2] == 2.0:
            SNLOS_loc.append(loc_PCP[ii, :2])
        else:
            print('wrong index')

    return LOS_loc, DNLOS_loc, SNLOS_loc

def PCP_plot(loc1, loc2, loc3):

    # import matplotlib.pyplot as plt

    xx1, yy1 = loc1[:, 0], loc1[:, 1]
    xx2, yy2 = loc2[:, 0], loc2[:, 1]
    xx3, yy3 = loc3[:, 0], loc3[:, 1]

    plt.scatter(xx1, yy1, marker='.', color='c') # , alpha=0.1
    plt.scatter(xx2, yy2, marker='.', color='y')
    plt.scatter(xx3, yy3, marker='.', color='k')

    plt.legend(['LOS', 'DNLOS', 'SNLOS'], loc="upper center", ncol=3)

    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.xlim([-100, 300])
    plt.ylim([-16, -9])
    plt.tight_layout()


def cal_acc(data1, data2):


    d2 = data2.flatten()
    N = d2.shape[0]
    d1 = data1.flatten()
    count = 0
    for kk in range(N):
        if int(d2[kk]) == int(d1[kk]):
            count = count + 1
    return count/N


def cal_mse(data1, data2):

    # data1 and data2 are M*2 numpy array matrix
    res_result = data1 - data2
    err = np.sqrt(res_result[:, 0] ** 2 + res_result[:, 1] ** 2)
    # print('size of err is', err.shape)
    return np.mean(err)

def result_analysis(true_result, est_result):


    from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

    # print("MSE is:\n", cal_mse(true_result, est_result))
    print("estimation error (RMSE) is:\n", np.sqrt(mean_squared_error(true_result, est_result))) #, squared=False  , squared=False  np.square(
    # print("estimation error (RMSE2) is:\n", mean_squared_error(true_result, est_result, squared=False))
    # err1 = (true_result - est_result)
    # err = np.sqrt(err1[:, 0]**2 + err1[:, 1]**2)



    err1 = (true_result - est_result)**2
    err = np.sqrt(err1[:, 0] + err1[:, 1])
    # err[err > 6] = 0
    # print('size of err is:', np.shape(np.array(err)))
    print("NP mean OF estimation error (RMSE) is:\n", np.mean(err))
    print('size of true_result is:\n', true_result.shape)

    # print("STD OF estimation error (RMSE) is:\n", np.std(err))  # , squared=False  , squared=False  np.square(
    # print("r2_score is:\n", r2_score(true_result, est_result))

    # fig4, ax4 = plt.subplots()
    # ax4.set_title('Hide Outlier Points')
    # ax4.boxplot(err, showfliers=False)


def result_analysis_OT(true_result, est_result, OT):

    err1 = (true_result - est_result)**2
    err = np.sqrt(err1[:, 0] + err1[:, 1])
    aa = int(err.shape[0] / OT)

    error_plots = np.zeros((OT, aa))

    for k in range(OT):
        error_p = err[k*aa:(k+1)*aa] # k*aa
        print("Mean OF {}-th estimation error (RMSE) is:\n", k, np.mean(error_p))
    # for k in range(OT):
    #     error_plots[k, :] = err[k*aa:(k+1)*aa]
    #     print("Mean OF {}-th estimation error (RMSE) is:\n", k, np.mean(err[k*aa:(k+1)*aa]))# k*aa
    #     # print("Variance OF {}-th estimation error (RMSE) is:\n", k, np.var(err[k * aa:(k + 1) * aa]))
    #
    # fig4, ax4 = plt.subplots()
    # # ax4.set_title('eee')
    # LABEL = ['T+10', 'T+20', 'T+30', 'T+40', 'T+50']
    # # LABEL = ['T+15', 'T+30', 'T+45']
    # ax4.boxplot(error_plots.T, labels=LABEL, showfliers=False, showmeans=True,  meanline=True) #, whis=0.5
    # ax4.set_ylabel('Root Mean Square Error(m)')
    # plt.ylim([-0.1, 3.5])
    # print("Mean OF ALL estimation error (RMSE) is:\n", np.mean(err))


# def result_analysis_OMT(true_result, est_result):
#
#     err1 = (true_result - est_result)**2
#     err = np.sqrt(err1[:, 0] + err1[:, 1])
#     aa = int(err.shape[0] / 5)
#
#     error_plots = np.zeros((5, aa))
#     methods = ['DCNN', 'TL', 'RDA', 'MDA', 'HDA']
#
#     for mm in methods:
#         saved_file = file0 + '/results/' + 'EXP2' + '/estimations/' + 'VP' + '/overtime/' + method + '/'
#         result0, result_est0 = np.load(saved_file + location), np.load(saved_file + location_est)
#
#         error_plots[k, :] = err[k*aa:(k+1)*aa]
#         print("Mean OF {}-th estimation error (RMSE) is:\n", k, np.mean(err[k*aa:(k+1)*aa]))
#         # print("Variance OF {}-th estimation error (RMSE) is:\n", k, np.var(err[k * aa:(k + 1) * aa]))
#
#     fig4, ax4 = plt.subplots()
#     # ax4.set_title('eee')
#     LABEL = ['T+10', 'T+20', 'T+30', 'T+40', 'T+50']
#     ax4.boxplot(error_plots.T, labels=LABEL, showfliers=False, showmeans=True,  meanline=True) #, whis=0.5
#     ax4.set_ylabel('Root Mean Square Error(m)')
#
#     print("Mean OF ALL estimation error (RMSE) is:\n", np.mean(err))


def variance_analysis(true_result, est_result, slices):

    std_in_slices =[]

    dif_result = np.sqrt((true_result - est_result) ** 2)
    err = np.sqrt(dif_result[:, 0] ** 2 + dif_result[:, 1] ** 2)
    # err[err > 6] = 0

    print('size of err is', err.shape)
    idx = err.shape[0]
    select_idx = int(idx / slices)
    for kk in range(slices):
        std_in_slices.append(np.std(err[kk*select_idx:(kk+1)*select_idx]))

    print('std_in_slices are \n', std_in_slices)
    all_std = np.std(np.array(std_in_slices))
    print('all_std are \n', all_std)
    return std_in_slices, all_std


def matrix16to64(old_matrix, p, q):

    # p, q dimenson of new matrix
    m, n = np.shape(old_matrix)

    matrix11 = np.zeros([int((p-m)/2), n])
    matrix13 = np.zeros([p, int((q - n) / 2)])
    temp1 = np.r_[old_matrix, matrix11]
    temp2 = np.r_[matrix11, temp1]
    temp3 = np.c_[matrix13, temp2]
    new_matrix = np.c_[temp3, matrix13]
    # print('dimension of new mtrix is ', new_matrix.shape)

    return new_matrix