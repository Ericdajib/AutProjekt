
# SPRING improved
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import time
import csv

# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph as pg

# define distance function
def dist_func(x, y):
    return abs(x - y)

# query sequence. Test
Q = np.array([1,2,3,2,1,2,3,4,5,3,2,1])
# Q = np.loadtxt("CMT_current.csv")
# Q = np.loadtxt("Pulse_current.csv")
# length of query sequence
m = len(Q)
# threshold for if we find the path
Threshold = 5

'''
 Define a STWM (subsequence time warping matrix) with (m+1) rows.
 The 0th row is 0. The m to (m+1) row of the 0th colum is infinity.
 Each cell of STWM contains two values: additive DTW distance and index of the first time of the matched subsequence.
 It is implemented by creating two matrices: STWM_D and STWM_I.
'''

# STWM_D for recording distance.
n = 3*m  # length of the column (start from index 1)
D = np.zeros([m+1,n+1]) # D is a matrix with m+1 rows and n+1 columns.
D[1:m+1,0] = np.inf # value of the index 0 is infinity.

# STWM_I for recording index.
I = np.zeros([m+1,n+1])

# define S_full for test.
S_full = np.array([1,2,3,2,1,2,3,4,5,3,2,1,7,7,12,15,4,3,8,14,5,1,2,3,2,1,2,3,4,5,6,3,2,1,1,0,4,2,3,6,7,8,2,2,3,2,3,2,3,4,1,2,3,2,1,2,2,3,4,5,6,3,2,1,0,8,10,3,4,5])
# S_full = np.loadtxt("TEST_current.csv")
# S_full = np.loadtxt("CMT-Puls_current.csv")
S = [] # S is timestream.
S_matched = 0
ax = []
ay = []
# from S_full generate timestream S.
# N: number of sampling
# st: value of current sampling
# t: time (read from sensor)
for N , st in enumerate(S_full):
    st = st
    S.append(st)
    time.sleep(0.1)
    N = N+1
    # t =

    # when SWTM is not full
    if N <= n:
        # STWM_D
        for i in range (1,m+1):
            D[i,N] = dist_func(Q[i-1],st) + min(D[i-1,N] , D[i,N-1] , D[i-1,N-1])

            # STWM_I
            if i == 1:
                I[i,N] = N
            else:
                if min(D[i-1,N] , D[i,N-1] , D[i-1,N-1]) == D[i-1,N-1]:
                    I[i,N] = I[i-1,N-1]
                elif min(D[i-1,N] , D[i,N-1] , D[i-1,N-1]) == D[i,N-1]:
                    I[i,N] = I[i,N-1]
                elif min(D[i-1,N] , D[i,N-1] , D[i-1,N-1]) == D[i-1,N]:
                    I[i,N] = I[i-1,N]

        # find path to match if D[m, N] less than threshold we set.
        if D[m, N] < Threshold:
            i = m
            j = N
            path = []
            count = 0
            S_matched = S[int(I[m, N])-1:N]

            while True:
                if i > 1 and j > I[m,N]:
                    path.append((i,j))
                    Min = min(D[i-1, j],D[i, j-1],D[i-1,j-1])

                    if Min == D[i - 1, j - 1]:
                        i = i - 1
                        j = j - 1
                        count = count + 1

                    elif Min == D[i, j - 1]:
                        j = j - 1
                        count = count + 1

                    elif Min == D[i - 1, j]:
                        i = i - 1
                        count = count + 1

                elif i == 1 and j == I[m,N]:    # If it goes to the bottom left, stop finding
                    path.append((i, j))
                    count = count + 1
                    break

                elif i == 1:    # If it goes to the bottom
                    path.append((i, j))
                    j = j - 1   # go to the left
                    count = count + 1

                elif j == I[m,N]:   # If it goes to the far left
                    path.append((i, j))
                    i = i - 1   # go to the bottom
                    count = count + 1
            print(path[::-1],count)

            print(S_matched)
        # print(D)
        # print(I)
        print("Abtast: % d" % N)
        print("Wert: % d" % st)
        # print(t)


        # plot real-time data
        plt.ion()
        plt.clf()

        # plot STWM_D
        D_plt = plt.subplot2grid((2,3),(0,1),colspan=2)
        D_plt.set_title("STWM_D")
        D_plt.imshow(D, origin='lower', cmap=plt.cm.binary, interpolation='nearest')

        if D[m, N] < Threshold:
            x_path, y_path = zip(*path)
            plt.plot(y_path, x_path)

        # plot query sequence
        Q_plt = plt.subplot2grid((2,3),(0,0),colspan=1)
        # base = plt.gca().transData
        # rot = transforms.Affine2D().rotate_deg(90)
        # plt.plot(Q,transform = rot+base)
        plt.plot(Q,color = "red")
        Q_plt.set_title("query sequence")

        # plot datastream
        S_plt = plt.subplot2grid((2,3),(1,1),colspan=2)
        ax.append(N)
        ay.append(st)
        plt.plot(ax, ay)
        # plt.xlim(N,N+3*m)
        S_plt.set_title("datastream")

        # plot matched sequence
        Sm_plt = plt.subplot2grid((2,3),(1,0),colspan=1)
        plt.plot(S_matched,color = "green")
        Sm_plt.set_title("matched sequence")

        plt.pause(0.002)
        plt.ioff()



    # when STWM is full
    if N > n:
        # STWM_D
        # roll the matrix,let the latest value in the last column.
        D = np.roll(D,-1,axis = 1)
        I = np.roll(I, -1, axis=1)
        for i in range (1,m+1):
            D[i,n] = dist_func(Q[i-1],st) + min(D[i-1,n], D[i,n-1] , D[i-1,n-1])

            # STWM_I
            if i == 1:
                I[i,n] = N
            else:
                if min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i-1,n-1]:
                    I[i,n] = I[i-1,n-1]
                elif min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i,n-1]:
                    I[i,n] = I[i,n-1]
                elif min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i-1,n]:
                    I[i,n] = I[i-1,n]

        # find path to match if D[m, n] less than threshold we set.
        if D[m,n] < Threshold:
            i = m
            j = n
            path = []
            count = 0
            S_matched = S[int(I[m, n])-1:N]

            while True:
                if i > 1 and j > (I[m, n] - (N-n)):
                    path.append((i, j))
                    Min = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

                    if Min == D[i - 1, j - 1]:
                        i = i - 1
                        j = j - 1
                        count = count + 1

                    elif Min == D[i, j - 1]:
                        j = j - 1
                        count = count + 1

                    elif Min == D[i - 1, j]:
                        i = i - 1
                        count = count + 1

                elif i == 1 and j == (I[m, n] - (N-n)): # If it goes to the bottom left, stop finding
                    path.append((i, j))
                    count = count + 1
                    break

                elif i == 1:    # If it goes to the bottom
                    path.append((i, j))
                    j = j - 1   # go to the left
                    count = count + 1

                elif j == (I[m, n] - (N-n)):    # If it goes the far left
                    path.append((i, j))
                    i = i - 1   # go to the bottom
                    count = count + 1
            print(path[::-1],count)
            print(S_matched)
        # print(D)
        # print(I)
        print("Abtast: % d" % N)
        print("Wert: % d" % st)
        # print(t)


        # plot real-time data
        plt.ion()
        plt.clf()

        # plot STWM_D
        D_plt = plt.subplot2grid((2,3),(0,1),colspan=2)
        D_plt.set_title("STWM_D")
        D_plt.imshow(D, origin='lower', cmap=plt.cm.binary, interpolation='nearest')
        if D[m, n] < Threshold:
            x_path, y_path = zip(*path)
            plt.plot(y_path, x_path)

        # plot query sequence
        Q_plt = plt.subplot2grid((2, 3), (0, 0), colspan=1)
        # base = plt.gca().transData
        # rot = transforms.Affine2D().rotate_deg(90)
        # plt.plot(Q, transform=rot + base)
        plt.plot(Q,color = "red")
        Q_plt.set_title("query sequence")

        # plot datastream
        S_plt = plt.subplot2grid((2, 3), (1, 1), colspan=2)
        ax.append(N)
        ay.append(st)
        plt.plot(ax, ay)
        # plt.xlim(N, N + 3 * m)
        S_plt.set_title("datastream")

        # plot matched sequence
        Sm_plt = plt.subplot2grid((2, 3), (1, 0), colspan=1)
        plt.plot(S_matched,color = "green")
        Sm_plt.set_title("matched sequence")

        plt.pause(0.002)
        plt.ioff()





















