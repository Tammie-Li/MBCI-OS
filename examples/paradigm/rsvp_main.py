import os
import warnings
import numpy as np
# from win32api import GetSystemMetrics
from psychopy import visual, core, parallel
# from algorithm.biosemi import ActiveTwo

warnings.filterwarnings('ignore')

# initial parallel port
# p_port = parallel.ParallelPort(address=0xAFF8)

# w = GetSystemMetrics(0)
# h = GetSystemMetrics(1)


def generate_texture(win, img_dir):
    img_num = len(os.listdir(img_dir))
    texture = []
    for _img in os.listdir(img_dir):
        img = os.path.join(img_dir, _img)
        texture_tmp = visual.ImageStim(win, image=img)
        texture.append(texture_tmp)
    return img_num, texture


# generate target and notarget textures
tar_cat_dir = 'stimulate/images/tar/cat'
tar_dog_dir = 'stimulate/images/tar/dog'
notar_dir = 'stimulate/images/notar'

# parameters
host = '10.170.33.219'
sfreq = 1024
port = 1111
channle_num = 65
trial_num = 2
seq_num = 50
notar_display_time = 0.2
tar_display_time = 0.15
tar_black_time = 0.05
duration = seq_num * notar_display_time + 0.5
tar_posi_cat = []
tar_posi_dog = []
data = []
label = []



def draw_fixation(win):
    length, width = 80, 3
    _w, _h = length / 1920, length / 1080
    l, r, u, d = (-_w, 0), (_w, 0), (0, -_h), (0, _h)
    line_lr = visual.Line(win, l, r, lineWidth=width)
    line_ud = visual.Line(win, u, d, lineWidth=width)
    line_lr.draw()
    line_ud.draw()
    win.flip()
    core.wait(2)


def get_tar_posi():
    tar_posi = np.zeros(4, dtype=int)
    tar_posi_cat = np.zeros(2, dtype=int)
    tar_posi_dog = np.zeros(2, dtype=int)
    flag = 0
    while(flag == 0):
        tar_list = np.random.permutation(30)
        for i in range(4):
            tar_posi[i] = tar_list[i] + 10
        tar_posi = sorted(tar_posi)
        flag = flag + 1
        for j in range(3):
            distance = tar_posi[j+1] - tar_posi[j]
            if distance <= 5:
                flag = 0
                break
        index = np.random.permutation(4)
        tar_posi_cat[0], tar_posi_cat[1] = tar_posi[index[0]], tar_posi[index[1]]
        tar_posi_dog[0], tar_posi_dog[1] = tar_posi[index[2]], tar_posi[index[3]]
        tar_posi_cat.sort()
        tar_posi_dog.sort()
    return tar_posi_cat, tar_posi_dog


def experience(index, path, name):
    print(index)
    w = 1920
    h = 1080

    win = visual.Window(size=(w, h), fullscr=True, color=(-1, -1, -1))
    # win = visual.Window(size=(w, h), fullscr=False, color=(-1, -1, -1))
    win.mouseVisible = False

    tar_cat_num, texture_tar_cat = generate_texture(win, tar_cat_dir)
    tar_dog_num, texture_tar_dog = generate_texture(win, tar_dog_dir)
    notar_num, texture_notar = generate_texture(win, notar_dir)
    

    for trial in range(trial_num):
        tar_posi_cat, tar_posi_dog = get_tar_posi()
        label.extend([tar_posi_cat, tar_posi_dog])
        draw_fixation(win)
        tar_rand_cat_idx = np.random.permutation(tar_cat_num)
        tar_rand_dog_idx = np.random.permutation(tar_dog_num)
        notar_rand_idx = np.random.permutation(notar_num)
        tar_count = 0
        notar_count = 0
    #     # active_two = ActiveTwo(host=host, sfreq=sfreq,
    #     #                        port=port, nchannels=channle_num)
        for seq in range(seq_num):
            if seq in tar_posi_cat:
                trigger = 4
                texture_tar_cat[tar_rand_cat_idx[tar_count]].draw()
                win.flip()
                core.wait(tar_display_time)
                win.flip()
                core.wait(tar_black_time)
                tar_count += 1
            elif seq in tar_posi_dog:
                trigger = 8
                texture_tar_dog[tar_rand_dog_idx[tar_count]].draw()
                win.flip()
                core.wait(tar_display_time)
                win.flip()
                core.wait(tar_black_time)
                tar_count += 1
            else:
                if seq in [0, 49]:
                    trigger = 16
                else:
                    trigger = 1
                texture_notar[notar_rand_idx[notar_count]].draw()
                core.wait(notar_display_time)
                win.flip()
                notar_count += 1
            # send trigger by parallel port
            # p_port.setData(trigger)

        win.flip()
    #     # read raw eeg data from BioSemi ActiveTwo device
    #     # raw_data = active_two.read(duration=duration)
    #     # data.append(raw_data)
    #     core.wait(1)
    win.close()
    data_path = path + '/' + name + '_data_' + str(index)
    label_path = path + '/' + name + '_label_' + str(index)
    np.save(data_path, np.array(data))
    np.save(label_path, np.array(label))



