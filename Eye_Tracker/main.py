from Lib.Tang2.device import emg_recorder_target
from multiprocessing import Queue, Process, Event
import numpy as np
import copy
import time


class EMGInterferce:
    def __init__(self):
        # 加载肌电采集系统参数
        parm = {'type':'s', 'param': {'port':'COM5','baudrate': 460800, 'devprocount':0}, 'devprocount':0}

        ctrparm = {}
        ctrparm['stopEv'] = Event()
        ctrparm['backQue'] = Queue()

        self.data_queue = Queue()
        emg_data_process = Process(target=emg_recorder_target, args=(parm, ctrparm, self.data_queue))
        emg_data_process.start()

        self.result = 0
        self.flag = False
    
    def emg_algorithm(self, x):
        # 输入为(C, T), Waveform Length (WL)
        features = []
        for chs in range(x.shape[0]):
            wl = np.sum(np.abs(np.diff(x[chs])))
            features.append(wl)
        features = np.array(features)
        print(np.mean(features))  # 修改这个阈值来调整肌电的灵敏度 
        if np.mean(features) > 15000:
            result = 1
        else:
            result = 0
        return result

    
    def start(self):
        self.flag = True
        while self.flag:
            if self.data_queue.empty(): continue
            emg_data = self.data_queue.get().T[:8, :]
            try:
                self.result = self.emg_algorithm(copy.deepcopy((emg_data)))
                print("当前系统预测的最新结果为：", self.result)
            except:
                print("推理过程有错误！！！")
            # time.sleep(0.025)

    def stop(self):
        self.flag = False


if __name__ == "__main__":
    emg = EMGInterferce()
    emg.start()
