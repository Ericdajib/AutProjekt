import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda,jit
import copy
import heapq as hq

threshold = -1
calculate_threshold = "y"
while calculate_threshold == "y":
    while True:
        calculate_threshold = input("\nFirstly you should get the threshold "
                                    "in order that this algorithm run successfully.\nCalculate threshold? (y/n)")
        if calculate_threshold not in ("y","n"):
            print("You can only type in y or n.\n")

        if calculate_threshold == "y":
            print("Calculating threshold...\n")
            break
        if calculate_threshold == "n":
            if threshold < 0:
                print("You have not calculate the threshold! \n")
            else:
                print("Run algorithm...\n")
                break

    Q = np.loadtxt("Pulse_current.csv")
    S_full = np.loadtxt("Puls_TEST_current.csv")
    # Q = np.array([1,2,3,2,1,2,3,4,5,3,2,1])
    # S_full = np.array([1,2,3,2,1,2,3,4,5,3,2,1,7,7,12,15,4,3,8,14,5,1,2,3,2,1,2,3,4,5,6,3,2,1,1,0,4,2,3,6,7,8,2,2,3,2,3,2,3,4,1,2,3,2,1,2,2,3,4,5,6,3,2,1,0,8,10,3,4,5])
    # threshold = 0
    m = len(Q)
    n = 2 * m
    S = np.zeros(n)
    D = np.zeros([m,n]); D[:,1] = np.inf
    I = np.zeros([m,n])
    ax = []
    ay = []
    D_cand_array = []
    S_matched_cand_array = []
    # S_matched = []
    additive_distance_sequence = []

    def plot():
        plt.ion()
        plt.clf()

        D_plt = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        D_plt.set_title("STWM_D")
        D_plt.imshow(D, origin='lower', cmap=plt.cm.binary, interpolation='nearest')

        if calculate_threshold == "y":
            I_plt = plt.subplot2grid((2, 3), (1, 1), colspan=2)
            I_plt.set_title("additive dtw distance")
            I_plt.plot(additive_distance_sequence)

        if calculate_threshold == "n":
            Sm_plt = plt.subplot2grid((2, 3), (1, 0), colspan=1)
            plt.plot(S_matched[::-1], color="green")
            Sm_plt.set_title("matched sequence")

            S_plt = plt.subplot2grid((2, 3), (1, 1), colspan=2)
            ax.append(N)
            ay.append(st)
            plt.plot(ax, ay)
            S_plt.set_title("datastream")

        Q_plt = plt.subplot2grid((2, 3), (0, 0), colspan=1)
        plt.plot(Q, color="red")
        Q_plt.set_title("query sequence")

        plt.pause(0.002)
        plt.ioff()

    @jit(nopython=True)
    def dist_func(x, y):
        return (x - y)**2


    @jit(nopython=True)
    def update_stwm(D_unupdated,I_unupdated,sampling,current_value):
        D = D_unupdated
        I = I_unupdated
        D[:, 1:] = D[:, :-1]
        I[:, 1:] = I[:, :-1]
        m = len(D)
        N = sampling
        st = current_value

        D[0, 0] = dist_func(Q[0], st)
        I[0, 0] = N
        for i in range(1, m):
            D[i, 0] = dist_func(Q[i], st) + min(D[i - 1, 1], D[i, 1], D[i - 1, 0])

            Min = min(D[i - 1, 1], D[i, 1], D[i - 1, 0])
            if Min == D[i - 1, 1]:
                I[i, 0] = I[i - 1, 1]
            elif Min == D[i, 1]:
                I[i, 0] = I[i, 1]
            elif Min == D[i - 1, 0]:
                I[i, 0] = I[i - 1, 0]

        return D, I


    def get_matched_sequence(stwm_d,stwm_i):
        global D_cand_array, S_matched_cand_array
        D = stwm_d
        I = stwm_i
        S_matched = []
        if D[-1, 0] <= threshold:
            if D[-1, 1] > threshold:
                D_cand_array = []
                S_matched_cand_array = []
            D_cand_array.append(D[-1, 0])
            S_matched_cand = S[0:int(N - I[-1, 0]) + 1]
            S_matched_cand_array.append(copy.copy(S_matched_cand))

        if D[-1, 0] > threshold:
            if D[-1, 1] <= threshold:
                D_cand_array = D_cand_array[::-1]
                local_min_index = len(D_cand_array) - 1 - D_cand_array.index(min(D_cand_array[::-1]))
                # local_min_index = D_cand_array.index(min(D_cand_array))
                S_matched = S_matched_cand_array[local_min_index]
                D_cand_array = []
                S_matched_cand_array = []
            if D[-1, 1] > threshold:
                D_cand_array = []
                S_matched_cand_array = []
        return S_matched



    N = 0
    while True:
        # time.sleep(1)
        st = S_full[N]
        S[1:] = S[:-1]
        S[0] = st

        D, I = update_stwm(D, I, N, st)
        if calculate_threshold == "y":
            additive_distance = D[-1, 0]
            additive_distance_sequence.append(additive_distance)
        else:
            S_matched = get_matched_sequence(D,I)

        if calculate_threshold == "n":
            plot()
        # print(f"sampling:{N+1}")
        # print(f"S_matched:{S_matched}")
        # print(f"dtw_dist:{D[-1,0]}")



        if N >= len(S_full) - 1:
            if calculate_threshold == "y":
                threshold = min(additive_distance_sequence)
                threshold = hq.nsmallest(int(math.ceil(len(additive_distance_sequence))/4),additive_distance_sequence)
                threshold = np.median(threshold)
                print(f"the threshold is: {threshold}.")
            break


        N = N + 1











