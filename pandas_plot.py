import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('auc_scores.pkl')

columns = [ 'googlenet_LIME',
 'googlenet_MP',
 'googlenet_RISE',
 'googlenet_SP',
 'googlenet_v2',
 'googlenet_v3',
 'googlenet_v4',
 'googlenet_v4_tv']

columnsedit = ['LIME', 'MP', 'RISE', 'SP', 'MIC', 'MIC_blur', 'MIC_inp', 'MIC_inp_tv']

dfmean = np.round(df[columns].mean().tolist(),3)
dfstd = np.round(df[columns].std().tolist(),3)
fig, ax = plt.subplots(figsize=(7,6))
xpos = np.arange(len(columns))
rects1 = ax.bar(xpos, dfmean, align='center', alpha=0.5, ecolor='black', capsize=10) #yerr=dfstd
rects1[4].set_color('g')
ax.set_xticks(xpos)
ax.set_xticklabels(columnsedit, rotation=45)
ax.legend()
ax.set_ylabel('Puntaje Borrado')
ax.set_title('Red GoogleNet - Dataset ImageNet (promedio de 1000 muestras)')
ax.yaxis.grid(True)

ax.bar_label(rects1, padding=3)

fig.tight_layout()

plt.show()

labels = ['LIME', 'MP', 'RISE', 'SP', 'MIC_tv']
iou = [0.37, 0.39, 0.43, 0.13, 0.51]
error = [0.52, 0.53, 0.36, 0.97, 0.18]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(7,6))
rects1 = ax.bar(x, iou, align='center', label='IoU', alpha=0.5, ecolor='black', capsize=10)
rects1[4].set_color('g')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('IoU')
ax.set_title('IoU Googlenet - Dataset COCO (promedio 480 muestras)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.yaxis.grid(True)

fig.tight_layout()

plt.show()
