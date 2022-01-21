
# SPRING improved
import numpy as np
import matplotlib.pyplot as plt
import time


# 定义差值
def dist_func(x, y):
    return abs(x - y)

# 确定查询序列，在此测试算法的序列为[1,2,3,2,1]
Q = np.array([1,2,3,2,1])
# Q的长度为m
m = len(Q)
Threshold = 1


# 定义一个可以更新的STWM矩阵，行数为m+1，第0行为0，第0列的1到m+1行为无限大
# STWM中保存累计DTW距离的STWM矩阵D
n = 40 # STWM中的列数（索引0不算）
D = np.zeros([m+1,n+1]) # D为m+1行，n+1列的矩阵
D[1:m+1,0] = np.inf # 第1个索引值到第m+1索引值是无限大

# STWM保存最短路径索引的矩阵I
I = np.zeros([m+1,n+1])

# 定义S总长度，用于测试
S_voll = np.array([1,2,3,2,1,3,4,5,4,3,1,2,3,3,2,1,3,4,3,4,3,1,2,3,2,2,1,1,0,0,6,1,2,3,2,1,3,4,5,6])
S = []

#生成已有的序列
for N , st in enumerate(S_voll):
    N = N+1
    st = st
    time.sleep(0.3)
    S.append(st)
    # t =


    # wenn STWM nicht voll
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

        # 小于阈值时，路径回溯
        if D[m, N] < Threshold:
            i = m
            j = N
            path = []
            count = 0

            while True:
                if i > 1 and j > I[m,N]:
                    path.append((i,j))
                    Min = min(D[i-1, j],D[i, j-1],D[i-1,j-1])

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

                elif i == 1 and j == I[m,N]:  # 如果走到最下角了
                    path.append((i, j))
                    count = count + 1
                    break

                elif i == 1:  # 如果走到最下边了
                    path.append((i, j))
                    j = j - 1  # 只能往左走
                    count = count + 1

                elif j == I[m,N]:  # 如果走到最左边了
                    path.append((i, j))
                    i = i - 1  # 只能往下走
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




















