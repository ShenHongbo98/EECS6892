import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

csv_data = pd.read_csv('G:\\files\\21Summer\\6892\\assignment2\\data.csv')  # 读取训练数据

Q_rewards = list(-csv_data['Q-learning'])
SARSA_rewards = list(-csv_data['SARSA'])

plt.plot(-np.log10(Q_rewards))
plt.plot(-np.log10(SARSA_rewards))
plt.legend(['Q-learning', 'SARSA'])
plt.xlabel('episode')
plt.ylabel('reward(-10^x)')
plt.show()
