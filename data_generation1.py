import os

import DeepMIMO
import numpy as np
import cmath, math


def generate_data(num_SC):
    # import DeepMIMO

    parameters = DeepMIMO.default_params()
    parameters['scenario'] = 'O2_dyn_3p4'  # O1_60   O2_dyn_3p5
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
    parameters['dynamic_settings']['last_scene'] = 1 + num_SC
    #

    parameters['activate_OFDM'] = 1  # Frequency (OFDM) or time domain channels
    parameters['num_paths'] = 25   # np.random.randint(10, 25)
    # parameters['num_paths'] = np.random.randint(10, 25)
    parameters['active_BS'] = np.array([1])  # np.array([1, 2])
    parameters['user_row_first'] = 1
    parameters['user_row_last'] = 31
    # parameters['row_subsampling'] = 0.25
    # parameters['user_subsampling'] = 0.25
    # parameters['enable_BS2BS'] = 1
    #

    parameters['OFDM']['subcarriers'] = 64
    parameters['OFDM']['subcarriers_limit'] = 64
    # parameters['OFDM']['subcarriers_sampling'] = 1
    # parameters['OFDM']['cyclic_prefix_ratio'] = 0.1
    parameters['OFDM']['bandwidth'] = 0.1
    # parameters['OFDM']['pulse_shaping'] = 0
    # parameters['OFDM']['rolloff_factor'] = 0.5
    # parameters['OFDM']['upsampling_factor'] = 50.0
    # parameters['OFDM']['low_pass_filter_ideal'] = 1
    #
    parameters['bs_antenna']['shape'] = np.array([1, 8, 8])
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

    for n in range(N):
        for nn in range(N):
            F[n, nn] = cmath.exp(-1j * 2 * math.pi * n * (nn - N / 2 ) / N)/np.sqrt(N)

    # U, V = np.eye(M), np.eye(N)

    return V, F


def index_sampling(sampling_ratio):
    if sampling_ratio == 1:
        sampled_index = [jj for jj in range(int(1891 * 31))]

    else:
        sampled_index = []
        cols = [ii for ii in range(31)]
        selected_cols = cols[::sampling_ratio]
        # for jj in range(int(31)):
        for jj in selected_cols:
            idx = np.array([kk for kk in range(1891)]) + int(jj * 1891)
            selected_idx = idx[::sampling_ratio]
            sampled_index.append(selected_idx)

        sampled_index = np.array(sampled_index)
        sampled_index = sampled_index.reshape(sampled_index.shape[0] * sampled_index.shape[1])

    return sampled_index


def loc_adjustment(locsX, min_X, min_Y):
    # aa, bb = locsX[:, 0], locsX[:, 1]

    aa, bb = [kk for kk in range(1891)], [kk for kk in range(31)]
    # tau_X = min_X + np.array(aa[::sampling_ratio]) * 0.2
    # tau_Y = min_Y + np.array(bb[::sampling_ratio]) * 0.2
    tau_X = min_X + np.array(aa) * 0.2
    tau_Y = min_Y + np.array(bb) * 0.2
    max_X, max_Y = max(tau_X), max(tau_Y)

    adj_loc = np.zeros(locsX.shape)
    adj_loc[:, 0] = locsX[:, 0] - ((max_X + min_X)) / 2
    adj_loc[:, 1] = locsX[:, 1] - ((max_Y + min_Y)) / 2

    # New_tau_X = tau_X - ((max_X + min_X)) / 2
    # New_tau_Y = tau_Y - ((max_Y + min_Y)) / 2

    # return adj_loc, New_tau_X, New_tau_Y
    return adj_loc


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    num_SC = 52
    test_SC = 4
    dataset = generate_data(num_SC)

    M = 64
    N = 64

    mode = 'UPA'
    U, V = generate_UV(M, N, mode)

    bs = 0

    selected_feature = 'adp'  # 'pdp' 'adp' 'rcsi' 'real_img'

    train_fingerprints, test_fingerprints = [], []
    train_loc, test_loc = [], []

    only_los, only_nlos = 0, 0

    for scene in range(num_SC):

        index_arr = dataset[scene][bs]['user']['LoS'][:]
        print("size of index_arr are:", len(index_arr), index_arr.shape)

        sampling_ratio = 4
        sampled_index_arr = index_sampling(sampling_ratio)

        all_loc0 = dataset[scene][bs]['user']['location'][:]
        # half_size = int(all_loc0.shape[0] / 2)
        # all_loc0 = all_loc0[:half_size, :]
        min_locX, min_locY = min(all_loc0[:, 0]), min(all_loc0[:, 1])

        # all_loc, t1, t2 = loc_adjustment(all_loc0, min_locX, min_locY)
        all_loc = loc_adjustment(all_loc0, min_locX, min_locY)

        indx_LOS = np.array(np.where(index_arr == 1))
        indx_NLOS = np.array(np.where(index_arr == 0))
        indx_CLOS = np.array(np.where(index_arr == -1))
        # print("len OF indx_LOS, indx_NLOS, and indx_CLOS are:{}/{}", indx_LOS.shape, indx_NLOS.shape, indx_CLOS.shape)


        loc_LOS = dataset[scene][bs]['user']['location'][indx_LOS]

        if scene == 0:
            selected_indices_los0 = set(sampled_index_arr).intersection(set(np.squeeze(indx_LOS)))
            selected_indices_los = np.array(list(selected_indices_los0))
        else:
            selected_indices_los = list(set(sampled_index_arr).intersection(set(np.squeeze(indx_LOS))))
            selected_indices_los = np.array(selected_indices_los)

        fingerprints_los = np.zeros((selected_indices_los.shape[0], M, N))
        for kk in range(selected_indices_los.shape[0]):
            id = selected_indices_los[kk]
            channel = np.squeeze(dataset[scene][bs]['user']['channel'][id])
            if selected_feature == 'adp':
                G = np.dot(np.dot(np.conj(U.T), channel), np.conj(V)) / np.sqrt(M * N)
                fingerprints_los[kk, :, :] = np.real(G * np.conj(G))
            elif selected_feature == 'pdp':
                pdp_channel = np.dot(np.conj(U.T), channel) / np.sqrt(M)
                fingerprints_los[kk, :, :] = np.abs(pdp_channel)
            elif selected_feature == 'rcsi':
                fingerprints_los[kk, :, :] = np.abs(channel)
            else:
                print('this type of FPs is not implemented')

        locations_los0 = all_loc[selected_indices_los, :]
        los_status = np.zeros((locations_los0.shape[0], 1))
        locations_los = np.concatenate((locations_los0[:, :2], los_status), axis=1)


        # selected_indices_nlos = list(set(sampled_index_arr).intersection(set(np.squeeze(indx_NLOS))))
        selected_indices_nlos = set(sampled_index_arr).intersection(set(np.squeeze(indx_NLOS)))


        # if scene == 0:
        #     s

        if scene > 0:
            selected_indices_dnlos = selected_indices_los0 & selected_indices_nlos
            selected_indices_snlos = selected_indices_dnlos ^ selected_indices_nlos
            selected_indices_dnlos = np.array(list(selected_indices_dnlos))
            selected_indices_snlos = np.array(list(selected_indices_snlos))

            locations_dnlos = np.zeros((selected_indices_dnlos.shape[0], 3))
            fingerprints_dnlos = np.zeros((selected_indices_dnlos.shape[0], M, N))

            fingerprints_snlos = np.zeros((selected_indices_snlos.shape[0], M, N))
            locations_snlos = np.zeros((selected_indices_snlos.shape[0], 3))

            for kk in range(selected_indices_dnlos.shape[0]):
                id = selected_indices_dnlos[kk]
                channel = np.squeeze(dataset[scene][bs]['user']['channel'][id])
                locations_dnlos[kk, :2] = all_loc[id, :2]
                locations_dnlos[kk, 2] = 1  # nlos_status = 1
                if selected_feature == 'adp':
                    G = np.dot(np.dot(np.conj(U.T), channel), np.conj(V)) / np.sqrt(M * N)
                    fingerprints_dnlos[kk, :, :] = np.real(G * np.conj(G))
                elif selected_feature == 'pdp':
                    pdp_channel = np.dot(np.conj(U.T), channel) / np.sqrt(M)
                    fingerprints_dnlos[kk, :, :] = np.abs(pdp_channel)
                elif selected_feature == 'rcsi':
                    fingerprints_dnlos[kk, :, :] = np.abs(channel)
                else:
                    print('this type of FPs is not implemented')

            for kk in range(selected_indices_snlos.shape[0]):
                id = selected_indices_snlos[kk]
                channel = np.squeeze(dataset[scene][bs]['user']['channel'][id])
                locations_snlos[kk, :2] = all_loc[id, :2]
                locations_snlos[kk, 2] = 2  # nlos_status = 2
                if selected_feature == 'adp':
                    G = np.dot(np.dot(np.conj(U.T), channel), np.conj(V)) / np.sqrt(M * N)
                    fingerprints_snlos[kk, :, :] = np.real(G * np.conj(G))
                elif selected_feature == 'pdp':
                    pdp_channel = np.dot(np.conj(U.T), channel) / np.sqrt(M)
                    fingerprints_snlos[kk, :, :] = np.abs(pdp_channel)
                elif selected_feature == 'rcsi':
                    fingerprints_snlos[kk, :, :] = np.abs(channel)
                else:
                    print('this type of FPs is not implemented')

            fingerprints_nlos = np.concatenate((fingerprints_dnlos, fingerprints_snlos))
            locations_nlos = np.concatenate((locations_dnlos, locations_snlos))

            fingerprints = np.concatenate((fingerprints_los, fingerprints_nlos))
            locations = np.concatenate((locations_los, locations_nlos))

        num_test_scene = test_SC

        print('current scene is:\n', scene)
        if scene == 0:
            continue
        elif scene == 1:
            train_fingerprints = fingerprints
            train_loc = locations # we do not need the reference data and it will be deleted

        elif (scene >= 2) and (scene < num_SC - num_test_scene) and (scene % 4 == 0):
            # print('SCENE IS:', scene)
            train_fingerprints = np.concatenate((train_fingerprints, fingerprints), axis=0)
            train_loc = np.concatenate((train_loc, locations), axis=0)
        # elif (scene == num_SC - num_test_scene) and scene%4 == 0:
        elif (scene == num_SC - num_test_scene):
            # print('SCENE IS:', scene)
            test_fingerprints = fingerprints
            test_loc = locations
        elif (scene < num_SC) and (scene % 4 == 0):
            # print('SCENE IS:', scene)
            test_fingerprints = np.concatenate((test_fingerprints, fingerprints), axis=0)
            test_loc = np.concatenate((test_loc, locations), axis=0)
        # else:


    print("sizes of train_fingerprints and test_fingerprints train_loc, test_loc are:{}/{}", train_fingerprints.shape,
          test_fingerprints.shape, train_loc.shape, test_loc.shape)

    # file = u'D:/researchAnnalRings/2022/coding/project1/data/trainANDtest/1m/fingerprints/classification/'
    file0 = os.getcwd()
    # file = file0 + '/results/EXP1/data3/'
    file = file0 + '/results/EXP1/DataDifSpeed/4X/'
    # file = file0 + '/results/EXP1/DataDifArea/difArea/'  # difBS # difArea
    # file = file0 + '/EXP2/data/VP/'  # /VP/
    # file = file0 + '/results/EXP6/data/RCSI/' # PDP
    # file = file0 + '/EXP5/16/data/PDP/'  # /RCSI/  ADP

    # print('sizes of train_fingerprints and test_loc are {} and {}', train_fingerprints.shape, test_loc.shape)

    saved1, saved2, saved3, saved4 = 'train_fingerprints', 'test_fingerprints', 'train_loc', 'test_loc'
    name1, name2, name3, name4 = 'train_fingerprints_2', 'test_fingerprints_2', 'train_loc_2', 'test_loc_2'
    # saved11, saved21, saved31, saved41 = 'train_fingerprints', 'test_fingerprints', 'train_loc', 'test_loc'

    np.save(file + saved1, train_fingerprints, allow_pickle=True)
    np.save(file + saved3, train_loc, allow_pickle=True)

    # np.save(file + saved2, test_fingerprints, allow_pickle=True)
    # np.save(file + saved4, test_loc, allow_pickle=True)

    # np.save(file + name1, train_fingerprints, allow_pickle=True)
    # np.save(file + name2, test_fingerprints, allow_pickle=True)
    # np.save(file + name3, train_loc, allow_pickle=True)
    # np.save(file + name4, test_loc, allow_pickle=True)

    #
    print(train_loc[-30:, :])

## save all files


    # from numpy import load
    #
    # trainF1 = load(file + saved1+'.npy')
    # testF1 = load(file + saved2+'.npy')
    # trainL1 = load(file + saved3+'.npy')
    # testL1 = load(file + saved4+'.npy')
    # trainF2 = load(file + name1+'.npy')
    # testF2 = load(file + name2+'.npy')
    # trainL2 = load(file + name3+'.npy')
    # testL2 = load(file + name4+'.npy')
    #
    # train_fingerprints = np.concatenate((trainF1, trainF2), axis=0)
    # test_fingerprints = np.concatenate((testF1, testF2), axis=0)
    # train_loc = np.concatenate((trainL1, trainL2), axis=0)
    # test_loc = np.concatenate((testL1, testL2), axis=0)
    #
    # np.save(file + saved11, train_fingerprints, allow_pickle=True)
    # np.save(file + saved21, test_fingerprints, allow_pickle=True)
    # np.save(file + saved31, train_loc, allow_pickle=True)
    # np.save(file + saved41, test_loc, allow_pickle=True)