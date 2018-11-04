import numpy as np
import matplotlib.pyplot as plt

N = 25

X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

def phiPoly(x, k):
    phi_ = []
    for i in range(len(x)):
        current = []
        for j in range(k + 1):
            curr = x[i] ** j
            current.append(float(curr))
        phi_.append(current)
    return phi_

def phiTrigo(x, k):
    phi_ = []
    for i in range(len(x)):
        current = [1]
        for j in range(1, k + 1):
            arg = 2 * j * np.pi * x[i]
            current.append(np.sin(arg))
            current.append(np.cos(arg))
        phi_.append(current)
    return phi_

def getWeight(phi, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)),np.transpose(phi)),y)

def plotPolynomial():
    plt.ylim(top=5)
    plt.ylim(bottom=-5)

    mockPoints = np.reshape(np.linspace(-0.3, 1.3, 200), (200, 1))
    plotMap = [(0, "red"), (1, "blue"), (2, "green"), (3, "cyan"), (11, "black")]
    for order, col in plotMap:
        weight = getWeight(phiPoly(X, order), Y)
        values = []
        for point in mockPoints:
            values.append(np.dot(phiPoly(point, order), weight)[0])
        plt.plot(mockPoints, values, color=col, label=("Degree " + str(order)))

    trueValues = np.cos(10*mockPoints**2) + 0.1 * np.sin(100*mockPoints)
    plt.scatter(mockPoints, trueValues, marker="P", label="original function")

    plt.legend()
    plt.show()

def plotTrigo():
    plt.ylim(top=5)
    plt.ylim(bottom=-5)

    mockPoints = np.reshape(np.linspace(-1, 1.2, 200), (200, 1))
    plotMap = [(1, "red"), (11, "green")]
    for order, col in plotMap:
        weight = getWeight(phiTrigo(X, order), Y)
        values = []
        for point in mockPoints:
            values.append(np.dot(phiTrigo(point, order), weight)[0])
        plt.plot(mockPoints, values, color=col, label="Order " + str(order))

    trueValues = np.cos(10*mockPoints**2) + 0.1 * np.sin(100*mockPoints)
    plt.scatter(mockPoints, trueValues, marker="P", label="original function")

    plt.legend()
    plt.show()

# plotPolynomial()
plotTrigo()

def getIntervals(N, K):
    intervalSize = int(N / K)
    start = 0
    intervals = []
    while start < N:
        end = min(start + intervalSize, N)
        intervals.append((start, end - 1))
        start = end

    print(intervals)
    return intervals

def crossValidation():
    mockPoints = np.reshape(np.linspace(-1, 1.2, 200), (200, 1))
    intervals = getIntervals(200, 10)
    trueValues = np.cos(10*mockPoints**2) + 0.1 * np.sin(100*mockPoints)

    order = 1
    for i in intervals:
        lefties = mockPoints[:(i[0] - 1)]
        lefties.reshape(1, len(lefties))
        righties = mockPoints[(i[1] + 1):]
        righties.reshape(1, len(righties))

        lefties_v = trueValues[:(i[0] + 1)]
        righties_v = trueValues[(i[1] + 1):]

        u = np.transpose(list(lefties) + list(righties))
        v = np.transpose(list(lefties_v) + list(righties_v))

        w = getWeight(phiTrigo(u, order), v)
        v = []
        for j in range(i[0], i[1] + 1):
            values.append(np.dot(phiTrigo(j, order), weight)[0])


# crossValidation()
