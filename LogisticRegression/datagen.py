import numpy as np
import matplotlib.pyplot as plt

def gen(n):
    np.random.seed(12)

    x1 = np.random.multivariate_normal([0,0], [[1, .75], [.75, 1]], n)
    x2 = np.random.multivariate_normal([1,4], [[1, .75], [.75, 1]], n)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(n),
                                  np.ones(n)))

    plt.figure(figsize=(12,8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c = simulated_labels, alpha = .4)
    plt.show()
    return [x1, x2]
