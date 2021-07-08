# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

y = [0.7063, 0.688992, 0.667662, 0.638535, 0.637535, 0.629184, 0.635976, 0.652806, 0.699021, 0.7011, 0.7210]

fig = plt.figure()

ax = fig.add_subplot(111)

plt.xlabel('α value')

plt.ylabel('MAE/RMSE')

h1=ax.plot(x,y,color='lightblue',marker='^',label='MAE')

h2=ax.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0.859367, 0.835, 0.811, 0.795973, 0.7901,  0.784602, 0.798365,  0.827168,  0.823537, 0.851908, 0.92034], color='darkgreen',marker='*',label='RMSE')

plt.legend()

ax.set_xlim(0,1)

#保存图片，名为’foo.png‘
plt.savefig('foo.png')

#显示所画的图，或者说运行
plt.show()