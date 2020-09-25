import matplotlib.pyplot as plt
import numpy as np

#singleThreadTime = 24*94.3
singleThreadTime = 24*21.8
threads = [1, 2, 4, 8, 12, 16, 20]
times = [(lambda x: singleThreadTime/x)(x) for x in threads]


plt.bar([(lambda x: str(x))(x) for x in threads], times)

plt.ylabel('Execution Time (Hours)')
plt.xlabel('Thread Count');

#plt.title('Plane Fit Statistics \n Execution Time')
plt.title('Mean & Standard Deviation \n Execution Time')
plt.show()
