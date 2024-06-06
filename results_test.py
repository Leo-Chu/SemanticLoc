
import numpy as np
import os
from utils import cdf_plot, result_plot, result_analysis, cal_acc, variance_analysis, result_analysis_OT
from functions import test_complexity


# EXPS = ['EXP3']  # , 'EXP2', 'EXP3' , 'EXP2', 'EXP3'  'EXP1', 'EXP2', 'EXP1',
# BWs = ['20M']  # ,
# methods = ['RDA']  # , 'HDA'  'DCNN', 'TL', 'RDA'  'DCNN', 'TL',
# paths = ['10PATH', 'VP']  # '25PATH'  , 'VP'

location, location_est = '_true_loc.npy', '_loc_est.npy'
sat, sat_est = '_true_sat.npy', '_sat_est.npy'

# input_norm = 'antenna_wise'   # , 'subcarrier_wise'
exp = 'EXP1X'
pp = '25PATH'   #   'VP' '10PATH'  '25PATH'
method = 'MDA' # 'DCNN'  #'TL' # 'RDA'  'MDA', 'HDA' 'MDA3'
bw = '20M'   #  '500M' '20M'
fp = 'ADP' # ['ADP', 'RCSI', 'PDP']
input_norm = 'no_norm'  # 'subcarrier_wise' 'antenna_wise' 'no_norm' 'default'
gamma = '0.3'


file0 = os.getcwd()

if exp == 'EXP0':
    saved_file = file0 + '/results/' + exp + '/estimations/' + gamma + '/'
elif exp == 'EXP1':
    if method == 'MDA2':
        saved_file = file0 + '/results/' + exp + '/estimations/' + method + '/' + str(gamma) + '/'
    elif method == 'MDA3':
        # saved_file = file0 + '/results/' + exp + '/estimations/' + method + '/' + str(gamma) + '/'
        saved_file = file0 + '/results/' + 'EXP1' + '/estimations/' + 'MDA3' + '/'
    else:
        saved_file = file0 + '/results/' + exp + '/estimations/' + method + '/'
elif exp == 'EXP1X':
    saved_file = file0 + '/results/EXP1/DataDifSpeed/2X/estimations/MDA/0.3/'
    # file0 + '/results/EXP1/DataDifSpeed/4X/estimations/MDA/0.3/'
    # file0 + '/results/EXP1/DataDifSpeed/2X/estimationsD/MDA/0.3/'

    # file0 + '/results/EXP1/DataDifArea/estimations/DCNN/' + str(0.3) + '/'
    # file0 + '/results/EXP1/DataDifArea/SAME/estimations/'
    # file0 + '/results/EXP1/DataDifArea/estimations/MDA/' + str(0.3) + '/'
    # file0 + '/results/EXP1/DataDifArea/SAME2/estimations/'


elif exp == 'EXP2':
    saved_file = file0 + '/results/' + exp + '/estimations/' + pp + '/' + method + '/'
elif exp == 'EXP2X':
    # saved_file = file0 + '/results/' + 'EXP2' + '/estimations/' + pp + '/' + method + '/'
    saved_file = file0 + '/results/' + 'EXP2' + '/estimations/' + pp + '/overtime/' + method + '/'
elif exp == 'EXP3':
    saved_file = file0 + '/results/' + exp + '/estimations/' + bw + '/' + method + '/'
elif exp == 'EXP4':
    saved_file = file0 + '/results/' + exp + '/estimations/' + input_norm + '/' + method + '/'
elif exp == 'EXP5':
    saved_file = file0 + '/results/' + exp + '/64' + '/estimations/' + fp + '/' + method + '/'
elif exp == 'EXP6':
    saved_file = file0 + '/results/' + exp + '/estimations/' + fp + '/' + method +  '/'
elif exp == 'EXP10':
    saved_file = file0 + '/results/' + exp + '/estimations/' + gamma + '/'
elif exp == 'EXP20':
    saved_file = file0 + '/results/' + exp + '/estimations/' + gamma + '/'
else:
    print('not implemented so far')

if exp == 'EXP0' or exp == 'EXP1' or exp == 'EXP10' or exp == 'EXP20':
    result0, result_est0 = np.load(saved_file+location), np.load(saved_file+location_est)
    result = np.reshape(result0, (result0.shape[0]*result0.shape[1], 2))
    result_est = np.reshape(result_est0, (result_est0.shape[0]*result_est0.shape[1], 2))
    LoS, LoS_est = np.squeeze(np.load(saved_file + sat)), np.load(saved_file + sat_est)

    # Los = np.array(LoS, dtype='int32')
else:
    result0, result_est0 = np.load(saved_file + location), np.load(saved_file + location_est)
    result = np.reshape(result0, (result0.shape[0] * result0.shape[1], 2))
    result_est = np.reshape(result_est0, (result_est0.shape[0] * result_est0.shape[1], 2))

# result0, result_est0 = np.load(saved_file+location), np.load(saved_file+location_est)
# result = np.reshape(result0, (result0.shape[0]*result0.shape[1], 2))
# result_est = np.reshape(result_est0, (result_est0.shape[0]*result_est0.shape[1], 2))


#

# print('size of result_est', result_est.shape)
#
print('---start-----result-----evaluation----')

if exp == 'EXP0' or exp == 'EXP1' or exp == 'EXP10' or exp == 'EXP20':
    print('+ are', LoS[:200])
    print('first LoS_est sample are', LoS_est[:200])
    print('blockage prediction accuracy is:', cal_acc(LoS, LoS_est))
    result_analysis(result, result_est)
    variance_analysis(result, result_est, 50)

elif exp == 'EXP2X':
    result_analysis_OT(result, result_est, 5)
else:
    # print('blockage prediction accuracy is:', cal_acc(LoS, LoS_est))
    result_analysis(result, result_est)







# result_analysis(result, result_est)
# result_plot(result, result_est)
# cdf_plot(result, result_est)


# from sklearn.metrics import mean_squared_error
# from utils import cal_mse
#
# print('skl mse', mean_squared_error(result, result_est))
# print("estimation error (RMSE) is:\n", mean_squared_error(result, result_est, squared=False)) #
# print('my mse', cal_mse(result, result_est))

# y_true = np.array([[0.5, 1],[-1, 1],[7, -6]])
# y_pred = np.array([[0, 2],[-1, 2],[8, -5]])
# print('skl mse', mean_squared_error(y_true, y_pred))
# print('my mse', cal_mse(y_true, y_pred))
