import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda,jit
import copy


Q = np.loadtxt("Pulse_current.csv")
S_full = np.loadtxt("Puls_TEST_current.csv")
# Q = np.array([1,2,3,2,1,2,3,4,5,3,2,1])
# S_full = np.array([1,2,3,2,1,2,3,4,5,3,2,1,7,7,12,15,4,3,8,14,5,1,2,3,2,1,2,3,4,5,6,3,2,1,1,0,4,2,3,6,7,8,2,2,3,2,3,2,3,4,1,2,3,2,1,2,2,3,4,5,6,3,2,1,0,8,10,3,4,5])
threshold = 1000
m = len(Q)
n = 2 * m
S = np.zeros(n)
D = np.zeros([m,n])
I = np.zeros([m,n])
I_range = 0
path = 0
S_matched = []
ax = []
ay = []

@jit(nopython=True)
def dist_func(x, y):
    return (x - y)**2


def plot():
    global D,I,m,ax,ay,S_matched
    plt.ion()
    plt.clf()

    # plot D
    D_plt = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    D_plt.set_title("STWM_D")
    D_plt.imshow(D, origin='lower', cmap=plt.cm.binary, interpolation='nearest')
    if D[m-1,0] <= threshold:
        x_path, y_path = zip(*path)
        plt.plot(y_path, x_path)


    # plot query sequence
    Q_plt = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    plt.plot(Q, color="red")
    Q_plt.set_title("query sequence")

    # plot STWM_I
    D_plt = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    D_plt.set_title("STWM_I")
    D_plt.imshow(I, origin='lower', cmap=plt.cm.binary, interpolation='nearest')

    # plot datastream
    S_plt = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    ax.append(N)
    ay.append(st)
    plt.plot(ax, ay)
    S_plt.set_title("datastream")

    # plot matched sequence
    Sm_plt = plt.subplot2grid((2, 3), (1, 0), colspan=1)
    plt.plot(S_matched[::-1], color="green")
    Sm_plt.set_title("matched sequence")


    plt.pause(0.002)
    plt.ioff()


# @cuda.jit
@jit(nopython=True)
def update_stwm(D_unupdated,I_unupdated,sampling,current_value):

    D = D_unupdated
    I = I_unupdated
    D[:, 1:] = D[:, :-1]
    I[:, 1:] = I[:, :-1]
    m = len(D)
    N = sampling
    st = current_value

    if N == 0:
        D[0, 0] = dist_func(Q[0], st)
        for i in range(1, m):
            D[i, 0] = dist_func(Q[i], st) + D[i - 1, 0]
        I[:, 0] = 0

    else:
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


def find_path(stwm_d, stwm_i):
    global S,N
    D = stwm_d
    I = stwm_i
    m = len(D)
    path = []
    I_range = int(N - I[m - 1, 0])
    S_matched_cand = S[0:I_range + 1]
    i = m - 1
    j = 0

    while True:
        if i > 0 and j < I_range:
            path.append((i, j))
            Min = min(D[i - 1, j + 1], D[i, j + 1], D[i - 1, j])
            if Min == D[i - 1, j + 1]:
                i = i - 1
                j = j + 1
                # count = count + 1
            elif Min == D[i, j + 1]:
                j = j + 1
                # count = count + 1
            elif Min == D[i - 1, j]:
                i = i - 1
                # count = count + 1

        elif i == 0 and j == I_range:
            path.append((i, j))
            # count = count + 1
            break
        elif i == 0:
            path.append((i, j))
            j = j + 1  # go to the left
            # count = count + 1
        elif j == I_range:  # If it goes to the far left
            path.append((i, j))
            i = i - 1  # go to the bottom
            # count = count + 1
    return S_matched_cand,path,I_range



N = 0
while True:
    # time.sleep(1)
    st = S_full[N]
    S[1:] = S[:-1]
    S[0] = st


    D,I = update_stwm(D,I,N,st)

    if N>0:
        if D[m-1,0] <= threshold:
            if D[m-1,1] > threshold:
                D_cand_array = []
                S_matched_cand_array = []
            D_cand_array.append(D[m - 1, 0])
            S_matched_cand, path, I_range = find_path(D,I)
            # S_matched_cand.tolist()
            S_matched_cand_array.append(copy.copy(S_matched_cand))

        if D[m - 1, 1] <= threshold and D[m - 1, 0] > threshold:
            D_cand_array = D_cand_array[::-1]
            local_min_index = len(D_cand_array) - 1 - D_cand_array.index(min(D_cand_array[::-1]))
            S_matched = S_matched_cand_array[local_min_index]
            D_cand_array = []


    # print(D,I)
    print(f"Abtast: {N + 1}")
    print(f"Wert: {st}")
    print(f"DTW_dist: {D[m-1,0]}\n")
    # print(f"range:{I_range}")
    # print(f"path:{path}")
    # print(f"sequence: {S}")
    # print(f"matched sequence: {S_matched}\n")

    plot()

    if N >= len(S_full) - 1:
        break
    N = N + 1













