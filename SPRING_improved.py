
# SPRING improved
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import kafka_test.consumer_test as kst
import kafka_test.producer_test as pst

# 定义差值
def dist_func(x, y):
    return abs(x - y)[0]

# 确定查询序列，在此测试算法的序列为[1,2,3,2,1]
Q = np.array([1,2,3,2,1])
# Q的长度为m
m = len(Q)
Threshold = 1



# 定义一个可以更新的STWM矩阵，行数为m+1，第0行为0，第0列的1到m+1行为无限大
# STWM中保存累计DTW距离的STWM矩阵D
n = 500 # STWM中的列数（索引0不算）
D = np.zeros([m+1,n+1]) # D为m+1行，n+1列的矩阵
D[1:m+1,0] = np.inf # 第1个索引值到第m+1索引值是无限大


# STWM保存最短路径索引的矩阵I
I = np.zeros([m+1,n+1])

while True:
    S =
    t =
    st =
    N =

    # wenn STWM nicht voll
    if N <= n:
        # STWM_D
        for i in range (1,m):
            D[i,N] = dist_func(Q[i-1],st) + min(D[i-1,N] , D[i,N-1] , D[i-1,N-1])

            # STWM_I
            if i == 1:
                I[i,N] == N
            else:
                if min(D[i-1,N] , D[i,N-1] , D[i-1,N-1]) == D[i-1,N-1]:
                    I[i,N] = N - 1
                elif min(D[i-1,N] , D[i,N-1] , D[i-1,N-1]) == D[i,N-1]:
                    I[i,N] = N - 1
                elif min(D[i-1,N] , D[i,N-1] , D[i-1,N-1]) == D[i,N]:
                    I[i,N] = N
        if D[m, N] < Threshold:




    # wenn STWM  voll
    if N > n:
        #STWM_D
        D = np.roll(D,-1,axis = 1)
        I = np.roll(I, -1, axis=1)
        for i in range (1,m):
            D[i,n] = dist_func(Q[i-1],st) + min(D[i-1,n], D[i,n-1] , D[i-1,n-1])

            # STWM_I
            if i == 1:
                I[i,n] == N
            else:
                if min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i-1,n-1]:
                    I[i,n] = N - 1
                elif min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i,n-1]:
                    I[i,n] = N - 1
                elif min(D[i-1,n], D[i,n-1] , D[i-1,n-1]) == D[i,n]:
                    I[i,n] = N

        if D[m,n] < Threshold:



















