

import numpy as np
import warnings
import scipy
import math
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from numpy import load
import torch
from torch import nn
from torch import optim
import math, sys
import argparse
from network_models0 import DR_net, Dataset
from network_models import TL_net, TL_net2, TL_net3
import os
from pathlib import Path
from utils import get_logger, cdf_plot, result_plot, result_analysis
import time
from torch.autograd import Variable


print(torch.cuda.is_available())
torch.cuda.current_device()
torch.cuda._initialized = True

def load_args():

    parser = argparse.ArgumentParser(description='localization code for IEEE JSAC by Leoski Chu')
    # parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="DeepMIMO")
    parser.add_argument('--dim', default=100, type=int, help='latent space size')
    parser.add_argument('--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=2048, type=int)
    parser.add_argument('--clz', default=28, type=int, help='number of investigated classes')
    parser.add_argument('--output_dim', default=4096, type=int)
    parser.add_argument('--sensors_channels', default=18, type=int)
    parser.add_argument('--slid_window_length', default=30, type=int)
    parser.add_argument('--dataset', default='P data')
    parser.add_argument('--grid_size', default='1m')
    parser.add_argument('--use_spectral_norm', default=True)
    parser.add_argument('--istrain', default=True)
    parser.add_argument('--samples_per_class', default=2000, type=float)
    parser.add_argument('--sampling_rate', default=120, type=float)
    parser.add_argument('--use_warm_start', default=False,
                        help='use warm start strategy to get good initial values for pseudo labels')
    parser.add_argument('--with_MI_regularization', default=False)  # True  False
    args = parser.parse_args()

    return args


def load_data(arg, logger):

    # \results\EXP1\data   train_fingerprints
    name1, name2, name3, name4 = 'train_fingerprints.npy', \
                                 'test_fingerprints.npy', \
                                 'train_loc.npy', \
                                 'test_loc.npy'

    file0 = arg.data_root0
    file1 = arg.data_root1
    print('Data loading from', file0)

    train_data0 = load(file0 + name1)
    test_data0 = load(file1 + name2)

    if arg.input_norm == 'default':
        max_train_adp = 0
        min_train_adp = 10
        max_train_adp1 = 0
        min_train_adp1 = 10
        for nn in range(train_data0.shape[0]):
            if np.linalg.norm(train_data0[nn, :, :]) > max_train_adp:
                max_train_adp = np.linalg.norm(train_data0[nn, :, :])
            if np.linalg.norm(train_data0[nn, :, :]) < min_train_adp:
                min_train_adp = np.linalg.norm(train_data0[nn, :, :])

        for nn in range(test_data0.shape[0]):
            if np.linalg.norm(test_data0[nn, :, :]) > max_train_adp1:
                max_train_adp1 = np.linalg.norm(test_data0[nn, :, :])
            if np.linalg.norm(test_data0[nn, :, :]) < min_train_adp1:
                min_train_adp1 = np.linalg.norm(test_data0[nn, :, :])

        # print("max_train_adp is", max_train_adp)
        # print("min_train_adp is", min_train_adp)
        logger.info("min_train_adp and min_train_adp are {} and {}".format(max_train_adp, min_train_adp))

        train_data0 = (np.array(train_data0) - min_train_adp) / (max_train_adp - min_train_adp)
        test_data0 = (np.array(test_data0) - min_train_adp1) / (max_train_adp1 - min_train_adp1)

    elif arg.input_norm == 'antenna_wise':
        max_train_adp, max_train_adp1 = np.zeros([1, 64]), np.zeros([1, 64])
        min_train_adp, min_train_adp1 = np.zeros([1, 64]), np.zeros([1, 64])
        temp1, temp2 = [], []
        for nn in range(int(train_data0.shape[0]/1)):
            temp1.append(np.squeeze(np.linalg.norm(train_data0[nn, :, :], axis=0, keepdims=True)))
        max_train_adp, min_train_adp = np.max(np.array(temp1), 0),  np.min(np.array(temp1), 0)
        train_data0 = (np.array(train_data0) - min_train_adp) / (max_train_adp - min_train_adp)
        test_data0 = (np.array(test_data0) - min_train_adp) / (max_train_adp - min_train_adp)

        for nn in range(int(test_data0.shape[0]/1)):
            temp2.append(np.squeeze(np.linalg.norm(test_data0[nn, :, :], axis=0, keepdims=True)))
        max_train_adp1, min_train_adp1 = np.max(np.array(temp2), 0),  np.min(np.array(temp2), 0)
        train_data0 = (np.array(train_data0) - min_train_adp) / (max_train_adp - min_train_adp)
        test_data0 = (np.array(test_data0) - min_train_adp1) / (max_train_adp1 - min_train_adp1)


    elif arg.input_norm == 'subcarrier_wise':
        max_train_adp = np.zeros([1, 64])
        min_train_adp = np.zeros([1, 64])
        temp1 = []
        for nn in range(int(train_data0.shape[0]/1)):
            temp1.append(np.squeeze(np.linalg.norm(train_data0[nn, :, :], axis=1, keepdims=True)))
        max_train_adp, min_train_adp = np.max(np.array(temp1), 0), np.min(np.array(temp1), 0)
        train_data0 = (np.array(train_data0) - min_train_adp) / (max_train_adp - min_train_adp)
        test_data0 = (np.array(test_data0) - min_train_adp) / (max_train_adp - min_train_adp)

    elif arg.input_norm == 'no_norm':
        train_data0 = np.array(train_data0)
        test_data0 = np.array(test_data0)
    else:
        print("data normalization scheme not considered")

    train_loc0 = load(file0 + name3)
    test_loc0 = load(file1 + name4)


    # train_data, test_data = train_data0[:90000, :, :], train_data0[90000:,:,:]
    # train_loc1, test_loc1 = train_loc0[:90000, :], train_loc0[90000:, :]


    # print('size of train_data0/test_data0 is:', train_data0.shape, test_data0.shape)

    scale, scale1 = 1/60, 1/60    # output normalization

    if arg.task == 'regression':
        train_loc = train_loc0[:, :2]
        train_loc0[:, 0] = scale*train_loc0[:, 0]
        test_loc = test_loc0[:, :2]
        test_loc0[:, 0] = scale*test_loc0[:, 0]

    elif arg.task == 'regressionX':
        train_loc0[:, 0] = scale * train_loc0[:, 0]
        bb = train_loc0[:, 2]
        train_loc0[:, 2] = bb.astype(int)
        train_loc = train_loc0
        test_loc0[:, 0] = scale1 * test_loc0[:, 0]
        test_loc0[:, 1] = scale1 * test_loc0[:, 1]
        aa = test_loc0[:, 2]
        test_loc0[:, 2] = aa.astype(int)
        test_loc = test_loc0

    else:
        print("Error task model !!!")

    train_data = Dataset(train_data0, train_loc)
    test_data = Dataset(test_data0, test_loc)

    print('Data loading finished!')

    return train_data, test_data





def net_training(train_dataset, test_dataset, arg, logger):

    print("The training starts from", time.asctime(time.localtime(time.time())))
    logger.info("The training starts from {}".format(time.asctime(time.localtime(time.time()))))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Current training is for method {}".format(arg.name))

    if arg.method == 'TL':
        nets = TL_net()
    elif arg.method == 'MDA' or arg.method == 'HDA':
        nets = TL_net2()
    elif arg.method == 'MDA2':
        nets = TL_net3().to(device)
    else:
        nets = DR_net()

    nets.to(device)

    if arg.method == 'HDA':
        log_var_a = torch.zeros((1,), requires_grad=True)
        log_var_b = torch.zeros((1,), requires_grad=True)
        params = ([p for p in nets.parameters()] + [log_var_a] + [log_var_b])
    else:
        params = nets.parameters()

    opt = optim.Adam(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.99, patience=10)
    # scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.97)
    loss_fn = torch.nn.MSELoss(size_average=True, reduce=True, reduction='mean')
    loss_BCE1 = torch.nn.CrossEntropyLoss()
    loss_BCE = torch.nn.NLLLoss()

    soure_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=arg.batch_size, drop_last=True, shuffle=True)
    target_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=arg.batch_size, drop_last=True, shuffle=True)

    nets.train()

    best_train_loss = arg.b_loss
    best_eval_loss = arg.b_loss/5
    train_loss = []
    eval_loss = []

    if arg.pretrained:
        # file0 = os.getcwd()
        # model_roots = file0 + '/results/' + 'EXP1' + '/saved_models/' + 'MDA' + '/'
        # model_path = Path(model_roots + '/' + arg.best_model + 'X.wgt')  # arg.method

        model_path = Path(arg.model_root + '/' + arg.best_model + 'X.wgt')  # arg.method
        nets.load_state_dict(torch.load(str(model_path)))

    for epoch in range(arg.epochs):
        if epoch % 100 == 0:
            print('Code is running and Current training epoch is', epoch)
        logger.info("Current training epoch is {}".format(epoch))

        soure_loader_iter, target_loader_iter = iter(soure_loader), iter(target_loader)
        len_dataloader = min(len(soure_loader), len(target_loader))

        epoch_acc, epoch_loss = 0, 0

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / arg.epochs / len_dataloader
            arg.alpha = 2. / (1. + np.exp(-10 * p)) - 1

            opt.zero_grad()

            csi_s, loc_s = next(soure_loader_iter)
            loc_s_LOC, loc_s_LOS = loc_s[:, :2],  loc_s[:, 2:]
            # print('size of are:{} and {}', loc_s_LOC.shape, loc_s_LOS.shape)

            csi_s, loc_s_LOC, loc_s_LOS = csi_s.to(device), loc_s_LOC.to(device), loc_s_LOS.to(device)

            if arg.method == 'HDA':
                log_var_a, log_var_b = log_var_a.to(device), log_var_b.to(device)

            domain_label = torch.zeros(arg.batch_size).long()
            domain_label = domain_label.to(device)

            if arg.method == 'DCNN':
                logit_output = nets(csi_s, arg)
                reg_error = loss_fn(loc_s.float(), logit_output.float())  # + 1e-3*para_l2_norm   pure regression

                all_loss = reg_error

            elif arg.method == 'TL':
                csi_t, loc_t = target_loader_iter.next()
                csi_t, loc_t = csi_t.to(device), loc_t.to(device)

                reg_output, _, dis_err = nets(csi_s, csi_t, arg)
                reg_error = loss_fn(loc_s.float(), reg_output.float())
                all_loss = reg_error + arg.alpha*dis_err

            elif arg.method == 'HDA':
                csi_t, loc_t = target_loader_iter.next()
                csi_t, loc_t = csi_t.to(device), loc_t.to(device)

                LR_output, _, PCP_output, _, dis_err = nets(csi_s, csi_t, arg)
                # print('size of PCP_output is:{} and {}', PCP_output.shape)
                cls_error = torch.sum(torch.exp(log_var_b) * loss_BCE1(torch.exp(log_var_b) * PCP_output, torch.squeeze(loc_s_LOS.long())) + log_var_b, -1)

                precision_a = torch.exp(log_var_a)
                diff_a = loss_fn(loc_s_LOC.float(), LR_output.float())
                reg_error = torch.sum(precision_a * diff_a + log_var_a, -1)
                all_loss = cls_error + 0.5*reg_error + arg.alpha*dis_err

            elif arg.method == 'MDA':
                csi_t, loc_t = next(target_loader_iter)
                csi_t, loc_t = csi_t.to(device), loc_t.to(device)

                LR_output, _, PCP_output, _, dis_err = nets(csi_s, csi_t, arg)
                # print('size of PCP_output is:{} and {}', PCP_output.shape)
                cls_error = loss_BCE1(PCP_output, torch.squeeze(loc_s_LOS.long()))
                reg_error = loss_fn(loc_s_LOC.float(), LR_output.float())
                all_loss = arg.gammar*cls_error + (1-arg.gammar)*reg_error + arg.alpha*dis_err

            elif arg.method == 'MDA2':
                csi_t, loc_t = next(target_loader_iter)
                csi_t, loc_t = csi_t.to(device), loc_t.to(device)


                LR_output, _, PCP_output, _, dis_err = nets(csi_s, csi_t, arg)
                # print('size of PCP_output is:{} and {}', PCP_output.shape)
                targets = torch.squeeze(loc_s_LOS.long())

                ptv = Variable(PCP_output.data.exp())
                #
                adaptive_output = (1-ptv) ** arg.temperature * PCP_output

                cls_error = loss_BCE(adaptive_output, targets)
                # cls_error = loss_BCE(PCP_output, targets)

                with torch.no_grad():
                    _, _, PCP_output, _, dis_err = nets(csi_s, csi_t, arg)

                PCP_output = PCP_output.detach()
                pt = PCP_output.data.exp()

                # print('size of pt is:{} and {}', pt.shape)

                reg_error = loss_fn(loc_s_LOC.float(), (1 + (1-pt)**arg.temperature)*LR_output.float())
                # reg_error = loss_fn(loc_s_LOC.float(), LR_output.float())
                all_loss = arg.gammar * cls_error + (1 - arg.gammar) * reg_error + arg.alpha * dis_err

            elif arg.method == 'RDA':

                logit_output, domain_output = nets(csi_s, arg)
                err_s_domain = loss_BCE(domain_output, domain_label)

                reg_error = loss_fn(loc_s.float(), logit_output.float())  # + 1e-3*para_l2_norm   pure regression

                csi_t, loc_t = target_loader_iter.next()
                csi_t, loc_t = csi_t.to(device), loc_t.to(device)

                domain_label = torch.ones(arg.batch_size).long()
                domain_label = domain_label.to(device)

                _, domain_output = nets(csi_t, arg)
                err_t_domain = loss_BCE(domain_output, domain_label)

                all_loss = reg_error + err_s_domain + err_t_domain
                # print('size of all_loss is', all_loss)

            all_loss.backward()
            opt.step()
            scheduler.step(all_loss)   #  KSE_loss: %f, mi_loss: %f, batch_train_loss: %f, batch_train_acc: %f'
            epoch_loss = epoch_loss + reg_error.cpu().detach()

            if arg.method == 'RDA':
                sys.stdout.write('\r [epoch: %d/ all %d], [iter: %d / all %d], reg_error: %f, err_s_domain: %f, err_t_domain: %f ' \
                                 % (epoch, arg.epochs, i + 1, len_dataloader, reg_error.data.cpu().numpy(),
                                    err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().numpy())) #
                # logger.info("Epoch {}, Iteration {}, len_dataloader = {}, reg_error = {}, err_s_domain = {}, err_t_domain ={}".format(epoch, i + 1,
                #                                       len_dataloader, reg_error.data.cpu().numpy(), err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().numpy()))

            elif arg.method == 'DCNN':
                sys.stdout.write(
                    '\r [epoch: %d/ all %d], [iter: %d / all %d], reg_error: %f' % (epoch, arg.epochs, i + 1, len(soure_loader), reg_error.data.cpu().numpy()))  #
                # logger.info("Epoch {}, Iteration {}, len_dataloader = {}, reg_error = {}".format(epoch, i + 1,
                #                                       len_dataloader, reg_error.data.cpu().numpy()))

            elif arg.method == 'TL' :
                sys.stdout.write('\r [epoch: %d/ all %d], [iter: %d / all %d], reg_error: %f, dis_err: %f ' % (epoch, arg.epochs, i + 1, len_dataloader, reg_error.data.cpu().numpy(), dis_err.data.cpu().numpy()))  #

            elif arg.method == 'MDA' or arg.method == 'HDA'or arg.method == 'MDA2':
                sys.stdout.write('\r [epoch: %d/ all %d], [iter: %d / all %d], reg_error: %f, los_error:%f, dis_err: %f ' % (epoch, arg.epochs, i + 1, len_dataloader, reg_error.data.cpu().numpy(),
                                                                                                                             cls_error.data.cpu().numpy(), dis_err.data.cpu().numpy()))  #
            else:
                print('wrong method')

            sys.stdout.flush()

        train_loss.append(epoch_loss/len(soure_loader))
        # print('\n The mean of train_loss is: ', np.mean(train_loss))
        logger.info("The mean of train_loss is: {}".format(np.mean(train_loss)))

        if np.mean(train_loss) < best_train_loss:
            best_train_loss = np.mean(train_loss)
            # print('minimum training best_train_loss is and happened at Epoch:', best_train_loss, epoch)
            logger.info("The mean of best_train_loss is: {} and happened at Epoch: {}".format(best_train_loss, epoch))
            enc_file = arg.model_root / Path(arg.best_model + '.wgt')  # + 'EPOCH' + str(epoch) + '_var_path_'
            enc_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(nets.state_dict(), str(enc_file))

            if epoch > 2:  # start to do evaluation after few epochs

                arg.test_mode = 'eval'
                evaluation_loss = test_model(test_dataset, arg, logger)
                # print('\n minimum evaluation_loss is and happened at Epoch:', evaluation_loss, epoch)

                if evaluation_loss < best_eval_loss:
                    best_eval_loss = evaluation_loss
                    # print('minimum best_eval_loss is and happened at Epoch:', best_eval_loss, epoch)
                    logger.info("minimum best_eval_loss is {} and happened at Epoch: {}".format(best_eval_loss, epoch))
                    enc_file = arg.model_root / Path(arg.best_model + 'X.wgt')  # + 'EPOCH' + str(epoch) + '_var_path_'
                    enc_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(nets.state_dict(), str(enc_file))

    print("The training ends at", time.asctime(time.localtime(time.time())))
    logger.info("The training ends at {}".format(time.asctime(time.localtime(time.time()))))


def test_model(test_dataset, arg, logger):

    from sklearn.metrics import mean_squared_error
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=arg.batch_size,
                                                drop_last=True, shuffle=True)

    if arg.method == 'TL':
        test_nets = TL_net().to(device)
    elif arg.method == 'MDA' or arg.method == 'HDA':
        test_nets = TL_net2().to(device)
    elif arg.method == 'MDA2':
        test_nets = TL_net3().to(device)
    else:
        test_nets = DR_net().to(device)

    if arg.test_mode == 'eval':
        model_path = Path(args.model_root + '/' + arg.best_model + '.wgt')
    else:
        model_path = Path(args.model_root + '/' + arg.best_model + 'X.wgt')
    print(model_path)
    test_nets.load_state_dict(torch.load(str(model_path)))
    test_nets.to(device)

    targetloc_est = []
    targetSAT_est = []
    true_loc = []
    true_sat = []
    test_nets.eval()
    arg.alpha = 0

    with torch.no_grad():
        target_loader_iter = iter(target_loader)
        for i in range(len(target_loader)):

            t_data, t_label = next(target_loader_iter)
            # t_data, t_label = t_data.to(device), t_label.to(device)

            t_label_LOC, t_label_LOS = t_label[:, :2], t_label[:, 2:]
            t_label_LOC, t_label_LOS = t_label_LOC.to(device), t_label_LOS.to(device)

            if arg.method == 'RDA':
                tloc_est, _ = test_nets(t_data, arg)
            elif arg.method == 'DCNN':
                tloc_est = test_nets(t_data, arg)
            elif arg.method == 'TL':
                tloc_est, _, _ = test_nets(t_data, t_data, arg)

            elif arg.method == 'MDA' or arg.method == 'HDA':
                tloc_est, _, tloc_SAT_est0, _, _ = test_nets(t_data, t_data, arg)
                _, tloc_SAT_est = torch.max(tloc_SAT_est0, 1)
            elif arg.method == 'MDA2':
                tloc_est0, _, tloc_SAT_est0, _, _ = test_nets(t_data, t_data, arg)
                _, tloc_SAT_est = torch.max(tloc_SAT_est0, 1)

                tloc_SAT0 = tloc_SAT_est0.detach()
                pt = tloc_SAT0.data.exp()
                tloc_est = (1 + (1 - pt) ** arg.temperature) * tloc_est0
            else:
                print('method not implemented')

            if arg.method == 'MDA' or arg.method == 'HDA' or arg.method == 'MDA2':
                targetloc_est.append(tloc_est.cpu().numpy())
                targetSAT_est.append(tloc_SAT_est.cpu().numpy())
                true_loc.append(t_label_LOC.cpu().numpy())
                true_sat.append(t_label_LOS.cpu().numpy())
            else:
                targetloc_est.append(tloc_est.cpu().numpy())
                true_loc.append(t_label.cpu().numpy())

    results_est, results_true = np.array(targetloc_est), np.array(true_loc)
    print('NMSE is', np.sqrt(mean_squared_error(results_true.flatten(), results_est.flatten())))

    if arg.test_mode == 'eval':
        # print('avg_error is', np.mean(avg_error))
        return np.sqrt(mean_squared_error(results_true.flatten(), results_est.flatten()))

    else:
        if arg.method == 'MDA' or arg.method == 'HDA' or arg.method == 'MDA2':
            np.save(arg.estimations_root + '_loc_est.npy', targetloc_est)
            np.save(arg.estimations_root + '_true_loc.npy', true_loc)
            np.save(arg.estimations_root + '_sat_est.npy', targetSAT_est)
            np.save(arg.estimations_root + '_true_sat.npy', true_sat)

        else:
            np.save(arg.estimations_root + '_loc_est.npy', targetloc_est)
            np.save(arg.estimations_root + '_true_loc.npy', true_loc)
        logger.info("Data test is good and ends at {}".format(time.asctime(time.localtime(time.time()))))


if __name__ == '__main__':

    args = load_args()
    args.scenarios = 'mixed'  #  'los' 'nlos' 'mixed'
    args.selected_feature = 'sample_cov'  # 'adp'
    args.task = 'regressionX'    # 'regression'   'classification'
    args.pretrained = False  # False True
    args.best_model = 'BestModel'
    args.b_loss = 100
    args.temperature = 2.0
    args.gammar = 0.3


### network training and test

    file0 = os.getcwd()

    args.method = 'MDA'

    args.data_root0 = file0 + '/results/' + 'EXP1' + '/data/'
    args.data_root1 = file0 + '/results/' + 'EXP1' + '/DataDifArea/'

    args.input_norm = 'antenna_wise'

    args.name = 'EXP1difArea' + '_' + args.input_norm + '_' + args.method
    logger = get_logger(args.name)
    train_dataset, test_dataset = load_data(args, logger)

    args.model_root = file0 + '/results/' + 'EXP1' + '/DataDifArea/' + '/saved_models/' + args.method + '/' + str(args.gammar) + '/'

    args.estimations_root = file0 + '/results/' + 'EXP1' + '/DataDifArea/' + '/estimations/' + args.method + '/' + str(args.gammar) + '/'

    net_training(train_dataset, test_dataset, args, logger)

    args.test_mode = 'test'
    test_model(test_dataset, args, logger)


