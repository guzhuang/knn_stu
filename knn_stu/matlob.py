import matplotlib
import knn_stu
import numpy as np
import matplotlib.pyplot as plt

datingDataMat,datingLabels = knn_stu.file2matrix('data/datingTestSet2.txt')

x = np.arange(6,1000)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
#ax1 = fig.add_subplot(221)
##ax1.plot(x ,x)

#ax2 = fig.add_subplot(111)
#ax2.plot(x ,1/(x**2))
plt.show()
