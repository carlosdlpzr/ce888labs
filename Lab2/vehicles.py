import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

vehicles = pd.read_csv('vehicles.csv')

plt.scatter(np.arange(vehicles['Current fleet'].count()), vehicles['Current fleet'], c='r')
plt.scatter(np.arange(vehicles['New Fleet'].count()), vehicles['New Fleet'].dropna(), c='b')
plt.savefig('scatterplot.png')
plt.show()


plt.hist(vehicles['Current fleet'])
plt.savefig('histCurrent.png')
plt.show()
plt.hist(vehicles['New Fleet'].dropna())
plt.savefig('histNew.png')
plt.show()
