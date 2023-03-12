from matplotlib import pyplot as plt    
import numpy as np

fig, ax = plt.subplots(figsize=(6.4, 3.2))
x = np.arange(3)
label = [2048, 4096, 8192]
y_up = [19.918, 79.103, 352.750]
y_dense = [63.517, 497.080, 4111.401]
y_dense_acc = [1,1,1]
y_up_acc = [0.9999922857280866, 0.9999939846955308, 0.9999944418880148]
width = 0.3
ax.set_xticks( x)
ax.set_xticklabels(label)
# ax.text(-1.1, 4150, 'Missed %:')
for i, acc in enumerate(y_up_acc):
#     ax.text(i-width-0.18, y_dense[i] + 10, 
# '''
# UP Time    Dense Time
# {}      {}
# {}% not missed
# '''.format(y_up[i], y_dense[i], round(acc* 100, 4) ), ha='left')
    ax.text(i-width/2, y_up[i] + 50, "{}s".format(round(y_up[i], 1)), ha = 'center')
    ax.text(i+width/2, y_dense[i] + 50, "{}s".format(round(y_dense[i], 1)), ha = 'center')
    ax.text(i, -1030, "Missed:\n{:.6f}%".format(round(1-y_up_acc[i],6)), ha = 'center')
ax.bar(x-width/2, y_up, width, label = 'UP')
ax.bar(x+width/2, y_dense, width, label = 'Dense')
ax.set_ylim(0, 4500)
# ax.set_xlim(-width-0.2, 2+width+0.3)
ax.legend()
# ax.set_xlabel("Resolution")
# ax.xaxis.set_label_coords(1, -0.03)
ax.set_ylabel("Time")
# ax2 = ax.twinx()
# ax2.plot(x, y_dense_acc)
# ax2.plot(x, y_up_acc)
fig.savefig('scal.pdf', bbox_inches='tight') 
fig.savefig('scal.png', bbox_inches='tight') 