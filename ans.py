import numpy as np
import matplotlib.pyplot as plt

N = 10000

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

    mockPoints = np.reshape(np.linspace(-0.3, 1.3, 1000), (1000, 1))
    plotMap = [(30, "black")]#[(0, "red"), (1, "blue"), (2, "green"), (3, "cyan"), (11, "black")]
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

plotPolynomial()
# plotTrigo()

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

def getStandardError(x, order):
    trueValues = np.cos(10*x**2) + 0.1 * np.sin(100*x)
    weight = getWeight(phiTrigo(x, order), trueValues)

    S = 0.
    for i in range(len(x)):
        value = np.dot(phiTrigo(x[i], order), weight)[0]
        diff = (value - trueValues[i]) ** 2
        S = S + diff
    return float(S/len(x))

def crossValidation(pointsN=1000, split=10, startP=0, endP=0.9):

    orders=range(0, 11)

    mockPoints = np.reshape(np.linspace(startP, endP, pointsN), (pointsN, 1))

    # np.random.shuffle(mockPoints)

    intervals = getIntervals(pointsN, split)
    trueValues = np.cos(10*mockPoints**2) + 0.1 * np.sin(100*mockPoints)

    averages = []
    standards = []
    for order in orders:
        errorsSum = 0.
        for i in intervals:
            lefties = mockPoints[:i[0]]
            righties = mockPoints[(i[1] + 1):]

            lefties_v = trueValues[:i[0]]
            righties_v = trueValues[(i[1] + 1):]

            u = np.concatenate([lefties, righties],axis=0)
            v = np.concatenate([lefties_v, righties_v],axis=0)

            weight = getWeight(phiTrigo(u, order), v)
            S = 0.
            curr = i[1] - i[0] + 1
            for j in range(i[0], i[1] + 1):
                value = np.dot(phiTrigo(mockPoints[j], order), weight)[0][0]
                expected = trueValues[j]
                S = S + ((value - expected) ** 2) / curr

            errorsSum += S

        standardError = getStandardError(mockPoints, order)
        standards.append(standardError)

        average = float(errorsSum / len(intervals))
        averages.append(average)

    plt.plot(orders, averages, label="Squared average error", color="blue")
    plt.plot(orders, standards, label="Standard error", color="red")
    plt.legend()
    plt.show()

# crossValidation()
