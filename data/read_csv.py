import csv
import numpy as np
import os

attachment_id = {}
type_id = {}
orientation = {}
surface = {}
startstage = {}
csv_file = csv.reader(open('label_infos.csv', 'r'))
for line in csv_file:
    attachment_id[line[0]] = [0 for i in range(128)]
    type_id[line[0]] = [0 for i in range(128)]
    orientation[line[0]] = [0 for i in range(128)]
    surface[line[0]] = [0 for i in range(128)]
    startstage[line[0]] = [-1 for i in range(128)]

csv_file = csv.reader(open('label_infos.csv', 'r'))
for line in csv_file:
    if line[0] != 'Name':
        if attachment_id[line[0]][int(line[1])*4] == 0:
            attachment_id[line[0]][int(line[1])*4] = int(line[2])
            type_id[line[0]][int(line[1])*4] = (line[3] == 'Optimized Rotation') + (line[3] == 'Rectangular')*2 + (line[3] == 'Ellipsoidal')*3
            orientation[line[0]][int(line[1])*4] = (line[4] == 'Vertical') + (line[4] == 'Horizontal') * 2
            surface[line[0]][int(line[1])*4] = (line[5] == 'Buccal') + (line[5] == 'Lingual')*2
            startstage[line[0]][int(line[1])*4] = int(line[6])
        elif attachment_id[line[0]][int(line[1])*4+1] == 0:
            attachment_id[line[0]][int(line[1])*4+1] = int(line[2])
            type_id[line[0]][int(line[1])*4+1] = (line[3] == 'Optimized Rotation') + (line[3] == 'Rectangular')*2 + (line[3] == 'Ellipsoidal')*3
            orientation[line[0]][int(line[1])*4+1] = (line[4] == 'Vertical') + (line[4] == 'Horizontal') * 2
            surface[line[0]][int(line[1])*4+1] = (line[5] == 'Buccal') + (line[5] == 'Lingual')*2
            startstage[line[0]][int(line[1])*4+1] = int(line[6])
        elif attachment_id[line[0]][int(line[1])*4+2] == 0:
            attachment_id[line[0]][int(line[1])*4+2] = int(line[2])
            type_id[line[0]][int(line[1])*4+2] = (line[3] == 'Optimized Rotation') + (line[3] == 'Rectangular')*2 + (line[3] == 'Ellipsoidal')*3
            orientation[line[0]][int(line[1])*4+2] = (line[4] == 'Vertical') + (line[4] == 'Horizontal') * 2
            surface[line[0]][int(line[1])*4+2] = (line[5] == 'Buccal') + (line[5] == 'Lingual')*2
            startstage[line[0]][int(line[1])*4+2] = int(line[6])
        elif attachment_id[line[0]][int(line[1])*4+3] == 0:
            attachment_id[line[0]][int(line[1])*4+3] = int(line[2])
            type_id[line[0]][int(line[1])*4+3] = (line[3] == 'Optimized Rotation') + (line[3] == 'Rectangular')*2 + (line[3] == 'Ellipsoidal')*3
            orientation[line[0]][int(line[1])*4+3] = (line[4] == 'Vertical') + (line[4] == 'Horizontal') * 2
            surface[line[0]][int(line[1])*4+3] = (line[5] == 'Buccal') + (line[5] == 'Lingual')*2
            startstage[line[0]][int(line[1])*4+3] = int(line[6])
try:
    os.mkdir('orientation')
    os.mkdir('attachment')
    os.mkdir('type')
    os.mkdir('surface')
    os.mkdir('startstage')
except Exception as e:
    pass
for name in attachment_id:
    np.save(os.path.join('attachment', name+'.npy'), np.array(attachment_id[name]))
    np.save(os.path.join('orientation', name+'.npy'), np.array(type_id[name]))
    np.save(os.path.join('type', name+'.npy'), np.array(type_id[name]))
    np.save(os.path.join('surface', name+'.npy'), np.array(surface[name]))
    np.save(os.path.join('startstage', name+'.npy'), np.array(startstage[name]))
    



