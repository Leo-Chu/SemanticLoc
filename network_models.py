

import torch # import main library
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torch.autograd import Function



class TL_net(nn.Module):  # transfer learning

    def __init__(self, input_channel=1, n_hiddenS=1024, n_hiddenR=512, n_filters=32, stride2D=1, clsR=8, clsR1=473):
        # def __init__(self, input_channel=1, n_hidden=2653, n_filters=32, stride2D=1, kernel_size=32, cls=2653):
        super(TL_net, self).__init__()

        self.distance_measure = nn.KLDivLoss(reduce=True)

        self.feature_extractor_s = nn.Sequential()
        self.feature_extractor_s.add_module('s_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_s.add_module('s_pol1', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act1', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_s.add_module('s_pol2', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act2', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_s.add_module('s_pol4', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act4', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_s.add_module('s_pol6', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act6', nn.ReLU(True))

        self.feature_extractor_t = nn.Sequential()
        self.feature_extractor_t.add_module('t_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_t.add_module('t_pol1', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act1', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_t.add_module('t_pol2', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act2', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_t.add_module('t_pol4', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act4', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_t.add_module('t_pol6', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act6', nn.ReLU(True))


        self.regression_s = nn.Sequential()
        self.regression_s.add_module('s_r_fc1', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_s.add_module('s_r_bn1', nn.BatchNorm1d(clsR+clsR1))
        self.regression_s.add_module('s_r_act1', nn.ReLU(True))
        self.regression_s.add_module('s_r_fc2', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_s.add_module('s_r_bn2', nn.BatchNorm1d(2*clsR))
        self.regression_s.add_module('s_r_act2', nn.ReLU(True))
        self.regression_s.add_module('s_r_fc3', nn.Linear(2*clsR, 2))

        self.regression_t = nn.Sequential()
        self.regression_t.add_module('t_r_fc1', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_t.add_module('t_r_bn1', nn.BatchNorm1d(clsR+clsR1))
        self.regression_t.add_module('t_r_act1', nn.ReLU(True))
        self.regression_t.add_module('t_r_fc2', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_t.add_module('t_r_bn2', nn.BatchNorm1d(2*clsR))
        self.regression_t.add_module('t_r_act2', nn.ReLU(True))
        self.regression_t.add_module('t_r_fc3', nn.Linear(2*clsR, 2))


    def cal_distance_error(self, x1, x2):

        log_probs1 = F.log_softmax(x1, 1)
        probs1 = F.softmax(x1, 1)
        log_probs2 = F.log_softmax(x2, 1)
        probs2 = F.softmax(x2, 1)
        Distance_estimate = (self.distance_measure(log_probs1, probs2) +
                             self.distance_measure(log_probs2, probs1)) / 2.0
        return Distance_estimate

    def forward(self, x_s, x_t, arg):

        x_s = x_s.type(torch.cuda.FloatTensor)
        x_s = x_s.view(-1, 1, 64, 64)
        x_t = x_t.type(torch.cuda.FloatTensor)
        x_t = x_t.view(-1, 1, 64, 64)

        Efeature_s = self.feature_extractor_s(x_s)
        Efeature_t = self.feature_extractor_t(x_t)

        # print('size of Efeature_s old is', Efeature_s.shape)
        Efeature_s = Efeature_s.view(arg.batch_size, -1)
        Efeature_t = Efeature_t.view(arg.batch_size, -1)
        # print('size of Efeature_s new is', Efeature_s.shape)

        est_s = self.regression_s(Efeature_s)
        est_t = self.regression_t(Efeature_t)

        dis_error = 0.9*self.cal_distance_error(Efeature_s, Efeature_t) + 0.1*self.cal_distance_error(est_s, est_t)

        return est_s, est_t, dis_error


class TL_net2(nn.Module):  # transfer learning

    def __init__(self, input_channel=1, n_hiddenS=1024, n_hiddenR=512, n_filters=32, stride2D=1, clsR=8, clsR1=473):
        # def __init__(self, input_channel=1, n_hidden=2653, n_filters=32, stride2D=1, kernel_size=32, cls=2653):
        super(TL_net2, self).__init__()

        self.distance_measure = nn.KLDivLoss(reduce=True)

        self.feature_extractor_s = nn.Sequential()
        self.feature_extractor_s.add_module('s_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_s.add_module('s_pol1', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act1', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_s.add_module('s_pol2', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act2', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_s.add_module('s_pol4', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act4', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_s.add_module('s_pol6', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act6', nn.ReLU(True))

        self.feature_extractor_t = nn.Sequential()
        self.feature_extractor_t.add_module('t_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_t.add_module('t_pol1', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act1', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_t.add_module('t_pol2', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act2', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_t.add_module('t_pol4', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act4', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_t.add_module('t_pol6', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act6', nn.ReLU(True))


        self.regression_s1 = nn.Sequential()
        self.regression_s1.add_module('s_r_fc11', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_s1.add_module('s_r_bn11', nn.BatchNorm1d(clsR+clsR1))
        self.regression_s1.add_module('s_r_act11', nn.ReLU(True))
        self.regression_s1.add_module('s_r_fc21', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_s1.add_module('s_r_bn21', nn.BatchNorm1d(2*clsR))
        self.regression_s1.add_module('s_r_act21', nn.ReLU(True))
        self.regression_s1.add_module('s_r_fc31', nn.Linear(2*clsR, 2))

        self.regression_s2 = nn.Sequential()
        self.regression_s2.add_module('s_r_fc12', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_s2.add_module('s_r_bn12', nn.BatchNorm1d(clsR+clsR1))
        self.regression_s2.add_module('s_r_act12', nn.ReLU(True))
        self.regression_s2.add_module('s_r_fc22', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_s2.add_module('s_r_bn22', nn.BatchNorm1d(2*clsR))
        self.regression_s2.add_module('s_r_act22', nn.ReLU(True))
        self.regression_s2.add_module('s_r_fc32', nn.Linear(2*clsR, 2))
        self.regression_s2.add_module('s_r_fc42', nn.Sigmoid())

        self.regression_t1 = nn.Sequential()
        self.regression_t1.add_module('t_r_fc11', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_t1.add_module('t_r_bn11', nn.BatchNorm1d(clsR+clsR1))
        self.regression_t1.add_module('t_r_act11', nn.ReLU(True))
        self.regression_t1.add_module('t_r_fc21', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_t1.add_module('t_r_bn21', nn.BatchNorm1d(2*clsR))
        self.regression_t1.add_module('t_r_act21', nn.ReLU(True))
        self.regression_t1.add_module('t_r_fc31', nn.Linear(2*clsR, 2))

        self.regression_t2 = nn.Sequential()
        self.regression_t2.add_module('t_r_fc12', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_t2.add_module('t_r_bn12', nn.BatchNorm1d(clsR+clsR1))
        self.regression_t2.add_module('t_r_act12', nn.ReLU(True))
        self.regression_t2.add_module('t_r_fc22', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_t2.add_module('t_r_bn22', nn.BatchNorm1d(2*clsR))
        self.regression_t2.add_module('t_r_act22', nn.ReLU(True))
        self.regression_t2.add_module('t_r_fc32', nn.Linear(2*clsR, 2))
        self.regression_t2.add_module('t_r_fc42', nn.Sigmoid())

    def cal_distance_error(self, x1, x2):

        log_probs1 = F.log_softmax(x1, 1)
        probs1 = F.softmax(x1, 1)
        log_probs2 = F.log_softmax(x2, 1)
        probs2 = F.softmax(x2, 1)
        Distance_estimate = (self.distance_measure(log_probs1, probs2) +
                             self.distance_measure(log_probs2, probs1)) / 2.0
        return Distance_estimate

    def forward(self, x_s, x_t, arg):

        x_s = x_s.type(torch.cuda.FloatTensor)
        x_s = x_s.view(-1, 1, 64, 64)
        x_t = x_t.type(torch.cuda.FloatTensor)
        x_t = x_t.view(-1, 1, 64, 64)

        Efeature_s = self.feature_extractor_s(x_s)
        Efeature_t = self.feature_extractor_t(x_t)

        # print('size of Efeature_s old is', Efeature_s.shape)
        Efeature_s = Efeature_s.view(arg.batch_size, -1)
        Efeature_t = Efeature_t.view(arg.batch_size, -1)
        # print('size of Efeature_s new is', Efeature_s.shape)

        est_s_loc = self.regression_s1(Efeature_s)
        est_t_loc = self.regression_t1(Efeature_t)
        est_s_state = self.regression_s2(Efeature_s)
        est_t_state = self.regression_t2(Efeature_t)

        dis_error = 0.8*self.cal_distance_error(Efeature_s, Efeature_t) + 0.1*self.cal_distance_error(est_s_loc, est_t_loc) + 0.1*self.cal_distance_error(est_s_state, est_t_state)

        return est_s_loc, est_t_loc, est_s_state, est_t_state, dis_error

class TL_net4(nn.Module):  # transfer learning

    def __init__(self, input_channel=1, n_hiddenS=1024, n_hiddenR=512, n_filters=32, stride2D=1, clsR=8, clsR1=473):
        # def __init__(self, input_channel=1, n_hidden=2653, n_filters=32, stride2D=1, kernel_size=32, cls=2653):
        super(TL_net4, self).__init__()

        self.distance_measure = nn.KLDivLoss(reduce=True)

        self.feature_extractor_s = nn.Sequential()
        self.feature_extractor_s.add_module('s_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_s.add_module('s_pol1', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act1', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_s.add_module('s_pol2', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act2', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_s.add_module('s_pol4', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act4', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_s.add_module('s_pol6', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act6', nn.ReLU(True))

        self.feature_extractor_t = nn.Sequential()
        self.feature_extractor_t.add_module('t_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_t.add_module('t_pol1', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act1', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_t.add_module('t_pol2', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act2', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_t.add_module('t_pol4', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act4', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_t.add_module('t_pol6', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act6', nn.ReLU(True))


        self.regression_s1 = nn.Sequential()
        self.regression_s1.add_module('s_r_fc11', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_s1.add_module('s_r_bn11', nn.BatchNorm1d(clsR+clsR1))
        self.regression_s1.add_module('s_r_act11', nn.ReLU(True))
        self.regression_s1.add_module('s_r_fc21', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_s1.add_module('s_r_bn21', nn.BatchNorm1d(2*clsR))
        self.regression_s1.add_module('s_r_act21', nn.ReLU(True))
        self.regression_s1.add_module('s_r_fc31', nn.Linear(2*clsR, 2))

        self.regression_s2 = nn.Sequential()
        self.regression_s2.add_module('s_r_fc12', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_s2.add_module('s_r_bn12', nn.BatchNorm1d(clsR+clsR1))
        self.regression_s2.add_module('s_r_act12', nn.ReLU(True))
        self.regression_s2.add_module('s_r_fc22', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_s2.add_module('s_r_bn22', nn.BatchNorm1d(2*clsR))
        self.regression_s2.add_module('s_r_act22', nn.ReLU(True))
        self.regression_s2.add_module('s_r_fc32', nn.Linear(2*clsR, 3))
        self.regression_s2.add_module('s_r_fc42', nn.Sigmoid())

        self.regression_t1 = nn.Sequential()
        self.regression_t1.add_module('t_r_fc11', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_t1.add_module('t_r_bn11', nn.BatchNorm1d(clsR+clsR1))
        self.regression_t1.add_module('t_r_act11', nn.ReLU(True))
        self.regression_t1.add_module('t_r_fc21', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_t1.add_module('t_r_bn21', nn.BatchNorm1d(2*clsR))
        self.regression_t1.add_module('t_r_act21', nn.ReLU(True))
        self.regression_t1.add_module('t_r_fc31', nn.Linear(2*clsR, 2))

        self.regression_t2 = nn.Sequential()
        self.regression_t2.add_module('t_r_fc12', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_t2.add_module('t_r_bn12', nn.BatchNorm1d(clsR+clsR1))
        self.regression_t2.add_module('t_r_act12', nn.ReLU(True))
        self.regression_t2.add_module('t_r_fc22', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_t2.add_module('t_r_bn22', nn.BatchNorm1d(2*clsR))
        self.regression_t2.add_module('t_r_act22', nn.ReLU(True))
        self.regression_t2.add_module('t_r_fc32', nn.Linear(2*clsR, 3))
        self.regression_t2.add_module('t_r_fc42', nn.Sigmoid())

    def cal_distance_error(self, x1, x2):

        log_probs1 = F.log_softmax(x1, 1)
        probs1 = F.softmax(x1, 1)
        log_probs2 = F.log_softmax(x2, 1)
        probs2 = F.softmax(x2, 1)
        Distance_estimate = (self.distance_measure(log_probs1, probs2) +
                             self.distance_measure(log_probs2, probs1)) / 2.0
        return Distance_estimate

    def forward(self, x_s, x_t, arg):

        x_s = x_s.type(torch.cuda.FloatTensor)
        x_s = x_s.view(-1, 1, 64, 64)
        x_t = x_t.type(torch.cuda.FloatTensor)
        x_t = x_t.view(-1, 1, 64, 64)

        Efeature_s = self.feature_extractor_s(x_s)
        Efeature_t = self.feature_extractor_t(x_t)

        # print('size of Efeature_s old is', Efeature_s.shape)
        Efeature_s = Efeature_s.view(arg.batch_size, -1)
        Efeature_t = Efeature_t.view(arg.batch_size, -1)
        # print('size of Efeature_s new is', Efeature_s.shape)

        est_s_loc = self.regression_s1(Efeature_s)
        est_t_loc = self.regression_t1(Efeature_t)
        est_s_state = self.regression_s2(Efeature_s)
        est_t_state = self.regression_t2(Efeature_t)

        dis_error = 0.8*self.cal_distance_error(Efeature_s, Efeature_t) + 0.1*self.cal_distance_error(est_s_loc, est_t_loc) + 0.1*self.cal_distance_error(est_s_state, est_t_state)

        return est_s_loc, est_t_loc, est_s_state, est_t_state, dis_error

class TL_net3(nn.Module):  # transfer learning

    def __init__(self, input_channel=1, n_hiddenS=1024, n_hiddenR=512, n_filters=32, stride2D=1, clsR=8, clsR1=473):
        # def __init__(self, input_channel=1, n_hidden=2653, n_filters=32, stride2D=1, kernel_size=32, cls=2653):
        super(TL_net3, self).__init__()

        self.distance_measure = nn.KLDivLoss(reduce=True)

        self.feature_extractor_s = nn.Sequential()
        self.feature_extractor_s.add_module('s_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_s.add_module('s_pol1', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act1', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_s.add_module('s_pol2', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act2', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_s.add_module('s_pol4', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act4', nn.ReLU(True))

        self.feature_extractor_s.add_module('s_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_s.add_module('s_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_s.add_module('s_pol6', nn.MaxPool2d(2))
        self.feature_extractor_s.add_module('s_act6', nn.ReLU(True))

        self.feature_extractor_t = nn.Sequential()
        self.feature_extractor_t.add_module('t_conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor_t.add_module('t_pol1', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act1', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor_t.add_module('t_pol2', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act2', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor_t.add_module('t_pol4', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act4', nn.ReLU(True))

        self.feature_extractor_t.add_module('t_conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor_t.add_module('t_bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor_t.add_module('t_pol6', nn.MaxPool2d(2))
        self.feature_extractor_t.add_module('t_act6', nn.ReLU(True))


        self.regression_s1 = nn.Sequential()
        self.regression_s1.add_module('s_r_fc11', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_s1.add_module('s_r_bn11', nn.BatchNorm1d(clsR+clsR1))
        self.regression_s1.add_module('s_r_act11', nn.ReLU(True))
        self.regression_s1.add_module('s_r_fc21', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_s1.add_module('s_r_bn21', nn.BatchNorm1d(2*clsR))
        self.regression_s1.add_module('s_r_act21', nn.ReLU(True))
        self.regression_s1.add_module('s_r_fc31', nn.Linear(2*clsR, 2))

        self.regression_s2 = nn.Sequential()
        self.regression_s2.add_module('s_r_fc12', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_s2.add_module('s_r_bn12', nn.BatchNorm1d(clsR+clsR1))
        self.regression_s2.add_module('s_r_act12', nn.ReLU(True))
        self.regression_s2.add_module('s_r_fc22', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_s2.add_module('s_r_bn22', nn.BatchNorm1d(2*clsR))
        self.regression_s2.add_module('s_r_act22', nn.ReLU(True))
        self.regression_s2.add_module('s_r_fc32', nn.Linear(2*clsR, 2))
        self.regression_s2.add_module('s_r_fc42', nn.LogSoftmax())

        self.regression_t1 = nn.Sequential()
        self.regression_t1.add_module('t_r_fc11', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_t1.add_module('t_r_bn11', nn.BatchNorm1d(clsR+clsR1))
        self.regression_t1.add_module('t_r_act11', nn.ReLU(True))
        self.regression_t1.add_module('t_r_fc21', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_t1.add_module('t_r_bn21', nn.BatchNorm1d(2*clsR))
        self.regression_t1.add_module('t_r_act21', nn.ReLU(True))
        self.regression_t1.add_module('t_r_fc31', nn.Linear(2*clsR, 2))

        self.regression_t2 = nn.Sequential()
        self.regression_t2.add_module('t_r_fc12', nn.Linear(n_hiddenR, clsR+clsR1))
        self.regression_t2.add_module('t_r_bn12', nn.BatchNorm1d(clsR+clsR1))
        self.regression_t2.add_module('t_r_act12', nn.ReLU(True))
        self.regression_t2.add_module('t_r_fc22', nn.Linear(clsR1+clsR, 2*clsR))
        self.regression_t2.add_module('t_r_bn22', nn.BatchNorm1d(2*clsR))
        self.regression_t2.add_module('t_r_act22', nn.ReLU(True))
        self.regression_t2.add_module('t_r_fc32', nn.Linear(2*clsR, 2))
        self.regression_t2.add_module('t_r_fc42', nn.LogSoftmax())

    def cal_distance_error(self, x1, x2):

        log_probs1 = F.log_softmax(x1, 1)
        probs1 = F.softmax(x1, 1)
        log_probs2 = F.log_softmax(x2, 1)
        probs2 = F.softmax(x2, 1)
        Distance_estimate = (self.distance_measure(log_probs1, probs2) +
                             self.distance_measure(log_probs2, probs1)) / 2.0
        return Distance_estimate

    def forward(self, x_s, x_t, arg):

        x_s = x_s.type(torch.cuda.FloatTensor)
        x_s = x_s.view(-1, 1, 64, 64)
        x_t = x_t.type(torch.cuda.FloatTensor)
        x_t = x_t.view(-1, 1, 64, 64)

        Efeature_s = self.feature_extractor_s(x_s)
        Efeature_t = self.feature_extractor_t(x_t)

        # print('size of Efeature_s old is', Efeature_s.shape)
        Efeature_s = Efeature_s.view(arg.batch_size, -1)
        Efeature_t = Efeature_t.view(arg.batch_size, -1)
        # print('size of Efeature_s new is', Efeature_s.shape)

        est_s_loc = self.regression_s1(Efeature_s)
        est_t_loc = self.regression_t1(Efeature_t)
        est_s_state = self.regression_s2(Efeature_s)
        est_t_state = self.regression_t2(Efeature_t)

        dis_error = 0.8*self.cal_distance_error(Efeature_s, Efeature_t) + 0.1*self.cal_distance_error(est_s_loc, est_t_loc) + 0.1*self.cal_distance_error(est_s_state, est_t_state)

        return est_s_loc, est_t_loc, est_s_state, est_t_state, dis_error


class soft_stair(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - temperature - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''

    # def __init__(self, in_features, temperature=None):
    def __init__(self, temperature=1e-4):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - temperature: trainable parameter
            temperature is initialized with zero value by default
        '''
        super(soft_stair, self).__init__()
        # self.in_features = in_features

        # initialize temperature
        if temperature == None:
            self.temperature = Parameter(torch.tensor(0.0))  # create a tensor out of temperature
        else:
            self.temperature = Parameter(torch.tensor(temperature))  # create a tensor out of temperature

        self.temperature.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x, arg):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''


        if arg.ROW ==1:
            Tau0 = torch.tensor(arg.Tau0)
            return torch.sum(Tau0 * torch.exp((x-Tau0).pow(2)/(-2*self.temperature))) / torch.sum(
                torch.exp((x-arg.Tau0).pow(2)/(-2*self.temperature)))
        else:
            Tau1 = torch.tensor(arg.Tau1)
            return torch.sum(Tau1 * torch.exp((x - Tau1).pow(2) / (-2 * self.temperature))) / torch.sum(
                torch.exp((x - Tau1).pow(2) / (-2 * self.temperature)))


        # if (self.temperature == 0.0):
        #     return x
        #
        # if (self.temperature < 0.0):
        #     return - torch.log(1 - self.temperature * (x + self.temperature)) / self.temperature
        #
        # if (self.temperature > 0.0):
        #     return (torch.exp(self.temperature * x) - 1) / self.temperature + self.temperature


# class Logger(object):
#     def __init__(self, fileN="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(fileN, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass