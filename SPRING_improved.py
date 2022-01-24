
# SPRING improved
import numpy as np
import matplotlib.pyplot as plt
import time


# define distance function
def dist_func(x, y):
    return abs(x - y)

# query sequence. Test:[1,2,3,2,1]
Q = np.array([1,2,3,2,1])
# length of query sequence
m = len(Q)
# threshold for if we find the path
Threshold = 1

'''
 Define a STWM (subsequence time warping matrix) with (m+1) rows.
 The 0th row is 0. The m to (m+1) row of the 0th colum is infinity.
 Each cell of STWM contains two values: additive DTW distance and index of the first time of the matched subsequence.
 It is implemented by creating two matrices: STWM_D and STWM_I
'''

# STWM_D for recording distance.
n = 40 # length of the column (start from index 1)
D = np.zeros([m+1,n+1]) # D is a matrix with m+1 rows and n+1 columns.
D[1:m+1,0] = np.inf # value of the index 0 is infinity.

# STWM_I for recording index.
I = np.zeros([m+1,n+1])

# define S_full for test.
S_full = np.array([1,2,3,2,1,3,4,5,4,3,1,2,3,3,2,1,3,4,3,4,3,1,2,3,2,2,1,1,0,0,6,1,2,3,2,1,3,4,5,6])
S = [] # S is timestream.

# from S_full generate timestream S.
# N: number of sampling
# st: value of current sampling
# t: time (read from sensor)
for N , st in enumerate(S_full):
    N = N+1
    st = st
    time.sleep(0.3)
    S.append(st)
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

                elif i == 1 and j == I[m,N]: # If it goes to the bottom left, stop finding
                    path.append((i, j))
                    count = count + 1
                    break

                elif i == 1:  # If it goes to the bottom
                    path.append((i, j))
                    j = j - 1  # go to the left
                    count = count + 1

                elif j == I[m,N]:  # If you go to the far left
                    path.append((i, j))
                    i = i - 1  # go to the bottom
                    count = count + 1
            print(path[::-1],count)

        plt.figure(figsize=(16,14))
        plt.subplot(4,1,1)
        plt.imshow(D, origin='lower', cmap=plt.cm.binary, interpolation='nearest')
        plt.title("STWM_D")
        plt.text(37, 8, str("Abtast:% d" % N))
        plt.text(37, 7, str("Wert vom Data Stream: % d" % st))
        if D[m, N] < Threshold:
            x_path, y_path = zip(*path)
            plt.plot(y_path, x_path, linewidth=5.0)

        plt.subplot(4,1,2)
        plt.imshow(I, origin='lower', cmap=plt.cm.binary, interpolation='nearest')
        plt.title("STWM_I")
        if D[m, N] < Threshold:
            x_path, y_path = zip(*path)
            plt.plot(y_path, x_path, linewidth=5.0)

        plt.subplot(4, 1, 3)
        plt.plot(Q, color='red')
        plt.title("Query Sequence")


        plt.subplot(4,1,4)
        plt.plot(np.array(S),color = 'blue')
        plt.title("Data Stream")

        plt.show()
        # x_path, y_path = zip(*path)
        # plt.plot(y_path, x_path)
        # plt.show()
        # print(I)

        print(D)
        print(I)
        print("Abtast: % d" % N )
        print("Wert: % d" % st)
            # print(t)





    # wenn STWM  voll
    if N > n:
        #STWM_D
        D = np.roll(D,-1,axis = 1)
        I = np.roll(I, -1, axis=1)
        for i in range (1,m):
            D[i,n] = dist_func(Q[i-1],st) + min(D[i-1,n], D[i,n-1] , D[i-1,n-1])

            # STWM_I
            if i == 1:
                I[i,n] = N
            else:
                if min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i-1,n-1]:
                    I[i,n] = N - 1
                elif min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i,n-1]:
                    I[i,n] = N - 1
                elif min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i,n]:
                    I[i,n] = N

        # 路径回溯
        if D[m,n] < Threshold:
            i = m
            j = n
            path = []
            count = 0

            while True:
                if i > 1 and j > (I[m, n] - (N-m)):
                    path.append((i, j))
                    Min = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

                    if Min == D[i - 1, j - 1]:  # 如果最小的点是左下角的点时
                        i = i - 1
                        j = j - 1
                        count = count + 1

                    elif Min == D[i, j - 1]:  # 如果最小的点是左边的点时
                        j = j - 1
                        count = count + 1

                    elif Min == D[i - 1, j]:  # 如果最小的点是下面的点时
                        i = i - 1
                        count = count + 1

                elif i == 1 and j == (I[m, n] - (N-m)):  # 如果走到最下角了
                    path.append((i, j))
                    count = count + 1
                    break

                elif i == 1:  # 如果走到最左边了
                    path.append((i, j))
                    j = j - 1  # 只能往下走
                    count = count + 1

                elif j == (I[m, n] - (N-m)):  # 如果走到最下边了
                    path.append((i, j))
                    i = i - 1
                    count = count + 1
            print(path[::-1])
        print (D, I, st, N)
        #print(t)




















