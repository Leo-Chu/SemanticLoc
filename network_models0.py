



import torch # import main library
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, temperature):
        ctx.temperature = temperature

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.temperature

        return output, None




class Dataset(torch.utils.data.Dataset):  # 需要继承data.Dataset

    def __init__(self, X, label):
        self.data_list = X
        self.data_label_list = label
        self.data_len = len(self.data_list)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return self.data_list[index], self.data_label_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.data_len


class DR_net(nn.Module):

    def __init__(self, input_channel=1, n_hiddenS=1024, n_hiddenR=512, n_filters=32, stride2D=1, kernel_size=32, clsR=2, clsS=2653):
        # def __init__(self, input_channel=1, n_hidden=2653, n_filters=32, stride2D=1, kernel_size=32, cls=2653):
        super(DR_net, self).__init__()


        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('conv1', nn.Conv2d(input_channel, n_filters, kernel_size=(6, 6), stride=stride2D))
        self.feature_extractor.add_module('bn1', nn.BatchNorm2d(1 * n_filters))
        self.feature_extractor.add_module('pol1', nn.MaxPool2d(2))
        self.feature_extractor.add_module('act1', nn.ReLU(True))

        self.feature_extractor.add_module('conv2', nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        self.feature_extractor.add_module('bn2', nn.BatchNorm2d(2 * n_filters))
        self.feature_extractor.add_module('pol2', nn.MaxPool2d(2))
        self.feature_extractor.add_module('act2', nn.ReLU(True))

        # self.feature_extractor.add_module('conv3', nn.Conv2d(2 * n_filters, 2 * n_filters, kernel_size=(4, 4), stride=stride2D))
        # self.feature_extractor.add_module('bn3', nn.BatchNorm2d(2 * n_filters))
        # self.feature_extractor.add_module('pol3', nn.MaxPool2d(2))
        # self.feature_extractor.add_module('act3', nn.ReLU(True))

        self.feature_extractor.add_module('conv4', nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=(3, 3), stride=stride2D))
        self.feature_extractor.add_module('bn4', nn.BatchNorm2d(3 * n_filters))
        self.feature_extractor.add_module('pol4', nn.MaxPool2d(2))
        self.feature_extractor.add_module('act4', nn.ReLU(True))

        # self.feature_extractor.add_module('conv5', nn.Conv2d(3 * n_filters, 3 * n_filters, kernel_size=(2, 2), stride=stride2D))
        # self.feature_extractor.add_module('bn5', nn.BatchNorm2d(3 * n_filters))
        # self.feature_extractor.add_module('pol5', nn.MaxPool2d(2))
        # self.feature_extractor.add_module('act5', nn.ReLU(True))

        self.feature_extractor.add_module('conv6', nn.Conv2d(3 * n_filters, 4 * n_filters, kernel_size=(2, 2), stride=stride2D))
        self.feature_extractor.add_module('bn6', nn.BatchNorm2d(4 * n_filters))
        self.feature_extractor.add_module('pol6', nn.MaxPool2d(2))
        self.feature_extractor.add_module('act6', nn.ReLU(True))

        self.regression = nn.Sequential()
        self.regression.add_module('r_fc1', nn.Linear(n_hiddenR, 128))
        self.regression.add_module('r_bn1', nn.BatchNorm1d(128))
        self.regression.add_module('r_act1', nn.ReLU(True))
        self.regression.add_module('r_fc2', nn.Linear(128, 16))
        self.regression.add_module('r_bn2', nn.BatchNorm1d(16))
        self.regression.add_module('r_act2', nn.ReLU(True))
        self.regression.add_module('r_fc3', nn.Linear(16, clsR))

        self.domain_classifier = nn.Sequential()

        self.domain_classifier.add_module('d_fc1', nn.Linear(n_hiddenR, 128))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(128))
        self.domain_classifier.add_module('d_act1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(128, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # self.fc1S = nn.Linear(n_hiddenS, n_hiddenS)
        # self.fc2S = nn.Linear(n_hiddenS, clsS)
        #
        # self.fc1R = nn.Linear(n_hiddenR, 16)
        # self.fc2R = nn.Linear(16, clsR)


    def forward(self, x, arg):

        x = x.type(torch.cuda.FloatTensor)
        x = x.view(-1, 1, 64, 64)
        Efeature = self.feature_extractor(x)
        Efeature = Efeature.view(arg.batch_size, -1)

        if arg.method == 'DCNN':
            reg_output = self.regression(Efeature)
            return reg_output

        elif arg.method == 'RDA':
            reg_output = self.regression(Efeature)
            reverse_Efeature = ReverseLayerF.apply(Efeature, arg.alpha)
            domain_output = self.domain_classifier(reverse_Efeature)
            return reg_output, domain_output

        else:
            print("Error task model")




# class dw_activation():   # direction_wise_activation
#
#
#     s
#
#
#
#
#     return s


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


