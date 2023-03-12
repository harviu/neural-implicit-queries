from matplotlib import pyplot as plt    
import numpy as np

fig, ax = plt.subplots(1, 2, sharex='all', figsize=(9, 4))
data_label = ['Vortex', 'Combustion', 'Ethanediol','Isotropic']

# for i,filename in enumerate([
#     'experiment_logs/hyper_vortex.log',
#     'experiment_logs/hyper_combustion.log',
#     'experiment_logs/hyper_ethane.log',
#     'experiment_logs/hyper_isotropic.log',
# ]):
for i,filename in enumerate([
    'experiment_logs/hyper_raua_vortex.log',
    'experiment_logs/hyper_raua_combustion.log',
    'experiment_logs/hyper_raua_ethane.log',
    'experiment_logs/hyper_raua_isotropic.log',
]):
    dense_time_sum = 0
    time = []
    iou = []
    f_score = []
    missed = []
    count = 0
    total_voxel = 0
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if '[Data]' in line:
                data = line[7:-1]
            elif '[Mode]' in line:
                mode = line[7:-1]
            elif '== Test' in line:
                count += 1
                while True:
                    if '[dense (GPU)]' in line:
                        dense_time_sum += float(line[23:28])
                    elif '[extract mesh]' in line:
                        time.append(float(line[24:29]))
                    elif '[Number missed]' in line:
                        missed.append(int(line[16:]))
                    elif '[Total true active voxel]' in line:
                        total_voxel = line[26:]
                    elif '[IoU]' in line:
                        iou.append(float(line[6:]))
                    elif '[Tree F-score]' in line:
                        f_score.append(float(line[15:]))
                    elif '=========================' in line:
                        break
                    line = f.readline()
            line = f.readline()

    dense_time = dense_time_sum/count
    time_percentage = np.asarray(time) / dense_time
    missed_percentage = 1-np.asarray(iou)
    # minimum = np.where(missed_percentage == 0, 1e10, missed_percentage).min()
    # maximum = missed_percentage.max()

    label = data_label[i]
    x = np.arange(len(f_score))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(x+1)
    ax[0].set_ylabel('Time Percentage')
    ax[0].plot(x, time_percentage, label = label, marker='.')
    # ax[0].plot(x, [dense_time] * len(time), label = label)
    ax[1].plot(x, missed_percentage, label = label, marker='.')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Log Missed Percentage')
    # ax[1].set_ylim(minimum)
    # ax[2].plot(x, f_score, label = label, marker='.')
    # ax[2].set_ylabel('F-score')
    print(data)
    z = 4
    print('Dense:', dense_time)
    print('Time:' ,time[z-1])
    print('Per_Time:',round(time_percentage[z-1] * 100,2))
    print("Missed: {:d}".format(missed[z-1]))
    print("Missed P: {:.6f}".format(missed_percentage[z-1]))
    print("F-score: {:.3f}".format(f_score[z-1]))


for aa in ax:
    aa.legend()
    aa.set_xlabel('Threshold z')

fig.savefig('hyper.png', bbox_inches='tight')
fig.savefig('hyper.pdf', bbox_inches='tight')


# for filename in [
#     'experiment_logs/hyper_raua_vortex.log',
#     'experiment_logs/hyper_raua_combustion.log',
#     'experiment_logs/hyper_raua_ethane.log',
#     'experiment_logs/hyper_raua_isotropic.log',
# ]: