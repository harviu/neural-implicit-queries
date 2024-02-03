from matplotlib import pyplot as plt    
import numpy as np

fig, ax = plt.subplots(2, 2, figsize=(7, 6))
data_label = ['Vortex', 'Ethanediol', 'Combustion','Isotropic']
data_name = ['vort', 'eth', 'comb' ,'iso']

for i in range(4):
    data = data_name[i]
    filename = 'experiment_logs/time_%s.log' % data
    modes = ['dense']
    modes_label = ['DENSE', 'UP-2', 'UP-5', 'UP-10', 'RAUA', 'FULL', 'FIXED', 'TRUN', 'APPE']
    times = {
        'ACP': [],
        'INR': [],
        'MC': [],
    }

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if '[Mode]' in line:
                mode = line[7:-1]
                modes.append(mode)
            elif '[dense with MC]' in line:
                dense_total_time = float(line[25:-8])
            elif 'Dense MC time is:' in line:
                dense_mc_time = float(line[18:])
                times['ACP'].append(0) 
                times['INR'].append(dense_total_time - dense_mc_time) 
                times['MC'].append(dense_mc_time)
            elif '== Test' in line:
                while True:
                    if '[	find nodes]' in line:
                        find_node_time = float(line[23:-8])
                    elif '[	query and mc]' in line:
                        query_mc_time = float(line[25:-8])
                    elif '[hierarchical with mc]' in line:
                        total_time = float(line[32:-8])
                    elif 'MC time is:' in line:
                        mc_time = float(line[12:])
                        times['ACP'].append(find_node_time) 
                        times['INR'].append(query_mc_time - mc_time) 
                        times['MC'].append(mc_time)
                        break
                    line = f.readline()
            line = f.readline()
    bottom = np.zeros(len(modes))
    axx = ax[i//2][i%2]
    axx.grid(axis = 'y')
    for boolean, time in times.items():
        p = axx.bar(modes_label, time, 0.5, label=boolean, bottom=bottom)
        bottom += time
    axx.set_title(data_label[i])
    axx.set_ylabel("Time (seconds)")
    axx.tick_params(axis='x', labelrotation=60)
    
ax[0][0].legend()
fig.tight_layout()
fig.savefig('time.pdf', bbox_inches='tight') 
fig.savefig('time.png', bbox_inches='tight') 
