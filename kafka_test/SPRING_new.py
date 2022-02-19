import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# Q = np.loadtxt("Pulse_current.csv")
# S_full = np.loadtxt("TEST_current.csv")
Q = np.array([1,2,3,2,1,2,3,4,5,3,2,1])
S_full = np.array([1,2,3,2,1,2,3,4,5,3,2,1,7,7,12,15,4,3,8,14,5,1,2,3,2,1,2,3,4,5,6,3,2,1,1,0,4,2,3,6,7,8,2,2,3,2,3,2,3,4,1,2,3,2,1,2,2,3,4,5,6,3,2,1,0,8,10,3,4,5])
threshold = 36500
m = len(Q)
n = 2 * m
S = np.zeros(m+1)
S_matched = np.array([])
D = np.zeros([m,n])
I = np.zeros([m,n])
# ax = np.array([])
# ay = np.array([])

@nb.jit(nopython=True)
def dist_func(x, y):
    return np.square(x - y)


def plot():
    global D,I,path,m
    plt.ion()
    plt.clf()

    # plot STWM_D
    D_plt = plt.subplot(2, 1, 1)
    D_plt.set_title("STWM_D")
    D_plt.imshow(D, origin='lower', cmap=plt.cm.binary, interpolation='nearest')
    if D[m-1,0] < threshold:
        x_path, y_path = zip(*path)
        plt.plot(y_path, x_path)

    # plot STWM_I
    I_plt = plt.subplot(2, 1, 2)
    I_plt.set_title("STWM_I")
    I_plt.imshow(I, origin='lower', cmap=plt.cm.binary, interpolation='nearest')

    plt.pause(0.002)
    plt.ioff()


N = 0
while True:
    st = S_full[N]
    S[1:] = S[:-1]
    S[0] = st
    D[:, 1:] = D[:, :-1]
    I[:, 1:] = I[:, :-1]

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

    # track the path
    if D[m-1,0] < threshold:
        I_range = int(N-I[m-1, 0])
        S_matched = S[0:I_range:-1]
        path = []
        count = 0
        i = m - 1
        j = 0

        while True:
            if i > 0 and j < I_range:
                path.append((i,j))
                Min = min(D[i-1, j+1], D[i,j+1], D[i-1,j])
                if Min == D[i-1, j+1]:
                    i = i-1
                    j = j+1
                    count = count+1
                elif Min == D[i, j+1]:
                    j = j+1
                    count = count+1
                elif Min == D[i-1,j]:
                    i = i-1
                    count = count+1

            elif i == 0 and j == I_range:
                path.append((i,j))
                count = count + 1
                break
            elif i == 0:
                path.append((i,j))
                j = j + 1  # go to the left
                count = count + 1
            elif j == I_range:  # If it goes to the far left
                path.append((i,j))
                i = i - 1  # go to the bottom
                count = count + 1

        print(path, count)
        print(S_matched[:-1])


    print(f"Abtast: {N+1}")
    print(f"Wert: {st}" )
    # print(D,I)

    plot()
    if N >= len(S_full)-1:
        break

    N = N+1











