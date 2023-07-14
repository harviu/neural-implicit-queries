from matplotlib import pyplot as plt    
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(1,2, figsize=(7.5, 3.2))
x = np.arange(3)
label = [2048, 4096, 8192]
y_up = [19.918, 79.103, 352.750]
y_dense = [63.517, 497.080, 4111.401]
y_dense_acc = [1,1,1]
y_up_acc = [0.9999922857280866, 0.9999939846955308, 0.9999944418880148]
width = 0.3

ax[0].set_xticks( x)
ax[0].set_xticklabels(label)
ax[0].grid(axis = 'y')
# ax.text(-1.1, 4150, 'Missed %:')
for i, acc in enumerate(y_up_acc):
#     ax.text(i-width-0.18, y_dense[i] + 10, 
# '''
# UP Time    Dense Time
# {}      {}
# {}% not missed
# '''.format(y_up[i], y_dense[i], round(acc* 100, 4) ), ha='left')
    ax[0].text(i-width/2, y_up[i] + 50, "{:.0f}s".format(round(y_up[i], 0)), ha = 'center')
    ax[0].text(i+width/2, y_dense[i] + 50, "{:.0f}s".format(round(y_dense[i], 0)), ha = 'center')
    # ax[0].text(i, -1030, "Missed:\n{:.6f}%".format(round(1-y_up_acc[i],6)), ha = 'center')
ax[0].bar(x-width/2, y_up, width, label = 'UP')
ax[0].bar(x+width/2, y_dense, width, label = 'Dense')
ax[0].set_ylim(0, 4500)
# ax.set_xlim(-width-0.2, 2+width+0.3)
ax[0].legend()
# ax.set_xlabel("Resolution")
# ax.xaxis.set_label_coords(1, -0.03)
# Create a formatter to display y-axis values in thousands
formatter = ticker.FuncFormatter(lambda x, pos: '{:.0f}k'.format(x * 1e-3))
# Apply the formatter to the y-axis tick labels
ax[0].yaxis.set_major_formatter(formatter)
ax[0].set_ylabel("Time (in thousand seconds)")
ax[0].set_xlabel("Target resolution")

y_missed = [1-x for x in y_up_acc]
ax2 = ax[0].twinx()
ax2.plot(x, y_missed, marker ='*', color='gray', label='Error')
ax2.set_ylabel('Missed Cells (%)') 
ax2.set_ylim(1e-6, 1e-5)
ax2.legend(loc='center left')

x = np.arange(8)

x_label = [
    3329, 6436, 12801,
    25021, 50177, 99732,
    198657, 396029,
]

dense_total_time = [
    0.210, 0.229, 0.245,
    0.296, 0.328, 0.493, 
    0.523, 0.924,
]
dense_mc_time = [
    0.008, 0.008, 0.010, 
    0.030, 0.029, 0.030, 
    0.030, 0.030
]
node_time = [
    0.072, 0.073, 0.076,
    0.082, 0.095, 0.137,
    0.179, 0.373,
]
query_time = [
    0.011, 0.016, 0.017,
    0.018, 0.021, 0.030,
    0.032, 0.057,
]
mc_time = [
    0.001, 0.001, 0.001,
    0.003, 0.002, 0.003,
    0.003, 0.003,
]
acc = [
    1.0, 0.999986973739058, 0.9999545800356546, 
    0.9999769143754185, 0.9997776418922675, 0.9998208065943174, 
    0.999555930041902, 0.999171019759255,
]
missed = [1-x for x in acc]
ax[1].set_xticks( x)
x_label = np.array(x_label)/1000
x_label = np.round(x_label,0)
x_label = ["%.0f" % x for x in x_label]
ax[1].set_xticklabels(x_label)
ax[1].grid(axis = 'y')

axx1 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
axx1.set_ylabel('Missed Cells (%)')  # we already handled the x-label with ax1
axx1.plot(x, missed, marker ='*', color='gray',label='Error')
axx1.legend(loc='center left')
# axx1.set_ylim(0,1e-3)

dense_query_time = [x-y for x, y in zip(dense_total_time, dense_mc_time)]

ax[1].bar(x-width/2, np.array(node_time)+np.array(mc_time)+np.array(query_time), width, label = 'UP', bottom=0 ) #, color='#1167b1')

ax[1].bar(x+width/2, dense_total_time, width, label = 'Dense', bottom=0) #, color='#ff781f')
# ax[1].bar(x+width/2, dense_mc_time, width, label = 'Dense MC', bottom=dense_query_time, color='#ff9d5c')

# ax[1].bar(x-width/2, node_time, width, label = 'UP APC', bottom=0, color='#1167b1')
# ax[1].bar(x-width/2, np.array(mc_time)+np.array(query_time), width, label = 'UP NIR', bottom=node_time, color='#00b4d8')
# ax[1].bar(x-width/2, mc_time, width, label = 'UP MC', bottom=np.array(node_time)+np.array(query_time), color='#187bcd')
# ax[1].text(-1, -0.3, "Missed:", ha = 'center', rotation=65)
# for i,a in enumerate(acc):
#     ax[1].text(i, -0.35, "{:.5f}%".format(round(1-a,5)), ha = 'center', rotation=65)

ax[1].set_ylabel("Time (seconds)")
ax[1].set_xlabel("# parameters (k)")

ax[1].legend()

fig.tight_layout()

fig.savefig('scal.pdf', bbox_inches='tight') 
fig.savefig('scal.png', bbox_inches='tight') 