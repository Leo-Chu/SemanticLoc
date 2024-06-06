
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#  see  https://deepmimo.net/versions/v2-python/ for more detailed introduction

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


import DeepMIMO
import numpy as np
import cmath, math
import matplotlib.pyplot as plt




# DEFAULT
# = {'dataset_folder': './Raytracing_scenarios',
#              'scenario': 'O1_60',
#              'dynamic_settings': {'first_scene': 1, 'last_scene': 5},
#              'num_paths': 10,
#              'active_BS': np.array([1, 2]),
#              'user_row_first': 1,
#              'user_row_last': 2,
#              'row_subsampling': 1,
#              'user_subsampling': 1,
#              'enable_BS2BS': 1,
#              'OFDM_channels': 1,
#              'OFDM': {'subcarriers': 512,
#               'subcarriers_limit': 32,
#               'subcarriers_sampling': 1,
#               'cyclic_prefix_ratio': 0.1,
#               'bandwidth': 0.05,
#               'pulse_shaping': 0,
#               'rolloff_factor': 0.5,
#               'upsampling_factor': 50.0,
#               'low_pass_filter_ideal': 1},
#              'bs_antenna': {'shape': np.array([ 1, 32,  1]), 'spacing': 0.5},
#              'ue_antenna': {'shape': np.array([1, 1, 1]), 'spacing': 0.5}
#              }




# Load the default parameters

def generate_data(num_SC):

    # import DeepMIMO

    parameters = DeepMIMO.default_params()
    parameters['scenario'] = 'O2_dyn_3p5'   # O1_60   O2_dyn_3p5
    parameters['dataset_folder'] = r'D:\researchAnnalRings\2022\coding\project1\data\scenarios'

    #### Set scenario parameters
    # parameters = {}
    # parameters['dynamic_settings'] = {}
    # parameters['OFDM'] = {}
    # parameters['bs_antenna'] = {}
    # parameters['ue_antenna'] = {}
    #
    # parameters['dataset_folder'] = './Raytracing_scenarios'
    # parameters['scenario'] = 'O1_60'
    parameters['dynamic_settings']['first_scene'] = 1
    parameters['dynamic_settings']['last_scene'] = num_SC
    #

    parameters['activate_OFDM'] = 1  # Frequency (OFDM) or time domain channels
    parameters['num_paths'] = 20
    # parameters['active_BS'] = np.array([1, 2])
    parameters['user_row_first'] = 1
    parameters['user_row_last'] = 31
    # parameters['row_subsampling'] = 1
    # parameters['user_subsampling'] = 1
    # parameters['enable_BS2BS'] = 1
    #

    parameters['OFDM']['subcarriers'] = 64
    parameters['OFDM']['subcarriers_limit'] = 64
    # parameters['OFDM']['subcarriers_sampling'] = 1
    # parameters['OFDM']['cyclic_prefix_ratio'] = 0.1
    parameters['OFDM']['bandwidth'] = 0.02
    # parameters['OFDM']['pulse_shaping'] = 0
    # parameters['OFDM']['rolloff_factor'] = 0.5
    # parameters['OFDM']['upsampling_factor'] = 50.0
    # parameters['OFDM']['low_pass_filter_ideal'] = 1
    #
    parameters['bs_antenna']['shape'] = np.array([1, 1, 64])
    parameters['bs_antenna']['spacing'] = 0.5

    parameters['ue_antenna']['shape'] = np.array([1, 1, 1])
    # parameters['ue_antenna']['spacing'] = 0.5


    # Generate data
    dataset = DeepMIMO.generate_data(parameters)

    return dataset

def generate_UV(M, N, anttena_pattern):
    # M, N = 64, 64

    # V = np.zeros((M, M)) + 1j*np.zeros((M, M))

    SM = round(np.sqrt(M))
    V = np.zeros((M, M), dtype=complex)
    F = np.zeros((N, N), dtype=complex)
    vv = np.zeros((SM, SM), dtype=complex)

    if anttena_pattern == 'linear':
        for k in range(M):
            for kk in range(M):
                V[k, kk] = cmath.exp(-1j * 2 * math.pi * k * (kk - M / 2) / M)/np.sqrt(M)  # - M / 2


    else:
        for k in range(SM):
            for kk in range(SM):
                vv[k, kk] = cmath.exp(-1j * 2 * math.pi * k * (kk - SM / 2) / SM)  # - M / 2

        V = np.kron(np.conj(vv.T), np.conj(vv.T))/np.sqrt(M)

    for k in range(N):
        for kk in range(N):
            F[k, kk] = cmath.exp(-1j * 2 * math.pi * k * kk / N)/np.sqrt(N)

    # U, V = np.eye(M), np.eye(N)

    return V, F


def plots_(matix1, matix2):

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #
    # # Make data.
    X = np.arange(0, 63, 1)
    Y = np.arange(0, 63, 1)
    X, Y = np.meshgrid(X, Y)
    Z = matix1[X, Y]
    #
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    #
    #
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.matshow(matix1)
    plt.show()


# def save_data(train_adp, test_adp, train_loc, test_loc):


    # file = u'D:/researchAnnalRings/2022/coding/project1/data/trainANDtest/'
    # np.save(file + saved1, train_adp, allow_pickle=True)
    # np.save(file + saved2, test_adp, allow_pickle=True)
    # np.save(file + saved3, train_loc, allow_pickle=True)
    # np.save(file + saved4, test_loc, allow_pickle=True)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    num_SC = 3
    dataset = generate_data(num_SC)



    M = 64
    N = 64

    U, V = generate_UV(M, N)

    bs = 1

    train_adp, test_adp = [], []
    train_loc, test_loc = [], []

    only_los, only_nlos = 0, 0

    for scene in range(num_SC):

        # scene = sce + 1
        # index_arr = dataset[:][:]['user']['LoS'][:]
        index_arr = dataset[scene][bs]['user']['LoS'][:]
        print("size of index_arr are:", index_arr.shape)

        indx_LOS = np.array(np.where(index_arr == 1))
        indx_NLOS = np.array(np.where(index_arr == 0))
        print("len OF indx_LOS and indx_NLOS are:{}/{}", indx_LOS.shape, indx_NLOS.shape)


        if only_los:
            saved1, saved2, saved3, saved4 = 'train_los_adp', 'test_los_adp', 'train_los_adp_loc', 'test_los_adp_loc'

            indices = indx_LOS
            loc_LOS = dataset[scene][bs]['user']['location'][indx_LOS]

            adp = np.zeros((loc_LOS.shape[1], M, N))
            locations = np.zeros((loc_LOS.shape[1], 3))

            for kk in range(loc_LOS.shape[1]):
                id = indices[:, kk]
                # print(id)
                channel = np.squeeze(dataset[scene][bs]['user']['channel'][id])
                # print(id)
                adp[kk, :, :] = np.abs(np.transpose(np.conj(U)) * channel * V)
                locations[kk, :] = dataset[scene][bs]['user']['location'][id]

        # del id
        elif only_nlos:

            saved1, saved2, saved3, saved4 = 'train_nlos_adp', 'test_nlos_adp', 'train_nlos_adp_loc', 'test_nlos_adp_loc'

            indices = indx_NLOS

            loc_NLOS = dataset[scene][bs]['user']['location'][indx_NLOS]

            adp = np.zeros((loc_NLOS.shape[1], M, N))
            locations = np.zeros((loc_NLOS.shape[1], 3))

            for kk in range(loc_NLOS.shape[1]):
                id = indices[:, kk]
                channel = np.squeeze(dataset[scene][bs]['user']['channel'][id])
                adp[kk, :, :] = np.abs(np.transpose(np.conj(U)) * channel * V)
                locations[kk, :] = dataset[scene][bs]['user']['location'][id]

        else:
            saved1, saved2, saved3, saved4 = 'train_mixed_adp', 'test_mixed_adp', 'train_mixed_adp_loc', 'test_mixed_adp_loc'

            indices = np.append(indx_LOS, indx_NLOS)
            print("len OF LOS, NLOS, and FULL INDEX are:{}/{}", indx_LOS.shape, indx_NLOS.shape, indices.shape)

            loc_LOS = dataset[scene][bs]['user']['location'][indx_LOS]
            loc_NLOS = dataset[scene][bs]['user']['location'][indx_NLOS]

            print("loc of los and nlos are:{}/{}", loc_LOS.shape, loc_NLOS.shape)

            adp = np.zeros((loc_LOS.shape[1]+loc_NLOS.shape[1], M, N))
            locations = np.zeros((loc_LOS.shape[1]+loc_NLOS.shape[1], 3))

            for kk in range(loc_LOS.shape[1]+loc_NLOS.shape[1]):
                id = indices[kk]
                channel = np.squeeze(dataset[scene][bs]['user']['channel'][id])
                adp[kk,:,:] = np.abs(np.transpose(np.conj(U)) * channel * V)
                locations[kk, :] = dataset[scene][bs]['user']['location'][id]

        num_test_scene = 1

        if scene==0:
            train_adp = adp
            train_loc = locations
        elif (scene>=1) and (scene< num_SC - num_test_scene):
            print(scene)
            train_adp = np.concatenate((train_adp, adp), axis=0)
            train_loc = np.concatenate((train_loc, locations), axis=0)
        else:
            test_adp = adp
            test_loc = locations
                # test_adp = np.concatenate((test_adp, adp), axis=0)
                # test_loc = np.concatenate((test_loc, locations), axis=0)

    # sample = 100
    # adp_sample = np.squeeze(adp[sample, :, :])
    #
    # plt.imshow(adp_sample)
    # plt.show()

    max_train_adp = 0
    for nn in range(train_adp.shape[0]):
        if np.linalg.norm(train_adp[nn,:,:])>max_train_adp:
            max_train_adp = np.linalg.norm(train_adp[nn,:,:])
    print("np.linalg.norm(train_adp[nn,:,:]) is", max_train_adp)

    train_adp = np.array(train_adp)/max_train_adp
    test_adp = np.array(test_adp)/max_train_adp
    train_loc = abs(np.array(train_loc))
    test_loc = abs(np.array(test_loc))

    print("sizes of train_adp and test_adp train_loc, test_loc are:{}/{}", train_adp.shape, test_adp.shape, train_loc.shape, test_loc.shape)

    file = u'D:/researchAnnalRings/2022/coding/project1/data/trainANDtest/adp/'
    np.save(file + saved1, train_adp, allow_pickle=True)
    np.save(file + saved2, test_adp, allow_pickle=True)
    np.save(file + saved3, train_loc, allow_pickle=True)
    np.save(file + saved4, test_loc, allow_pickle=True)


    # print("max_train_adp is", max())
    # print(train_loc[-30:, :])


### plots

    # ii = 1
    # id = int(np.linspace(0,train_loc.shape[0], ii))
    # plt.figure()
    # line, = plt.plot(id, np.squeeze(train_loc[id, ii]), lw=2)
    # plt.show()