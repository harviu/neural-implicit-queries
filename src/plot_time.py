from matplotlib import pyplot as plt    
import numpy as np

fig, ax = plt.subplots(2, 2, figsize=(9, 8))
data_label = ['Vortex', 'Ethanedioal', 'Combustion','Isotropic']
data_name = ['vort', 'eth', 'comb' ,'iso']

for i in range(4):
    data = data_name[i]
    filename = 'experiment_logs/time_%s.log' % data
    modes = ['dense']
    modes_label = ['DENSE', 'UP', 'RAUA', 'ALL', 'FIXED', 'TRUN', 'APPEN']
    times = {
        'RA': [],
        'NIR': [],
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
                times['RA'].append(0) 
                times['NIR'].append(dense_total_time - dense_mc_time) 
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
                        times['RA'].append(find_node_time) 
                        times['NIR'].append(query_mc_time - mc_time) 
                        times['MC'].append(mc_time)
                        break
                    line = f.readline()
            line = f.readline()
    bottom = np.zeros(len(modes))
    axx = ax[i//2][i%2]
    for boolean, time in times.items():
        p = axx.bar(modes_label, time, 0.5, label=boolean, bottom=bottom)
        bottom += time
    axx.set_title(data_label[i])
    axx.set_ylabel("Time (seconds)")
    
ax[0][0].legend()

fig.savefig('time.pdf', bbox_inches='tight') 
fig.savefig('time.png', bbox_inches='tight') 
