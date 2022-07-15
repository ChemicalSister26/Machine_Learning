import numpy as np
import matplotlib.pyplot as plt

N = 100
sigma = 3
k = 0.5
b = 2

f = np.array([k*i+b for i in range(N)])
y = f + np.random.normal(0, sigma, N)

x = np.array(range(N))
mx = x.sum()/N
my = y.sum()/N
a2 = np.dot(x.T, x)/N
a1 = np.dot(x.T, y)/N


k1 = (a1 - mx*my)/(a2 - mx**2)
b1 = my - k1*mx
f1 = np.array([k1*i+b1 for i in range(N)])

plt.plot(f, c='green')
plt.plot(f1)
plt.scatter(x, y, s=2, c='red')
plt.grid(True)
plt.show()
