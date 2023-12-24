import json
import os
import numpy as np
import matplotlib.pyplot as plt


with open('running_time/no_multi.json') as f:
        no_multi = json.load(f)

no_multi = np.array(no_multi)


result_files = [s for s in os.listdir('running_time') if s.endswith('.json') and s[:2] != 'no']
vis_results = {}

for result_file in result_files:
    dir_result_file = os.path.join('running_time/', result_file)
    with open(dir_result_file) as f:
        module_result = json.load(f)
        module_result = np.array(module_result)

        speed_up = no_multi/module_result
        avr_speed_up = np.mean(speed_up)

        vis_results[result_file.split('.')[0]] = (avr_speed_up, speed_up, module_result, np.mean(module_result))

colors = ['r', 'g', 'b']
for ind, module in enumerate(vis_results.keys()):
     time_per_image = vis_results[module][2]
     avr_time = vis_results[module][3]
     plt.plot(np.arange(len(time_per_image)), time_per_image, label=module, linestyle='dashed', c=colors[ind])
     plt.plot(np.arange(len(time_per_image)), [avr_time]*len(time_per_image), label='average '+module, c=colors[ind])
     plt.legend()
     plt.title('Time spend per image')
     plt.xlabel('image number')
     plt.ylabel('time')

plt.savefig('running_time/time_per_image.png')
plt.clf()




for ind, module in enumerate(vis_results.keys()):
     all_speed_up = vis_results[module][1]
     avr_speed_up = vis_results[module][0]
     plt.plot(np.arange(len(all_speed_up)), all_speed_up, label=module, c=colors[ind], linestyle='dashed')
     plt.plot(np.arange(len(all_speed_up)), [avr_speed_up]*len(all_speed_up), label='average '+module, c=colors[ind])
     plt.legend()
     plt.title('Speed up for all images')
     plt.xlabel('image number')
     plt.ylabel('Speed up for one image')

plt.savefig('running_time/all_images_speed_up.png')
plt.clf()