import numpy as np
import data as mydata

model = mydata.TimeSeriesSampler(5, mydata.sample_func1)

x = model.sample(2000)

# Plot

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.kdeplot(x, label='KDE of Sample Data')
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()