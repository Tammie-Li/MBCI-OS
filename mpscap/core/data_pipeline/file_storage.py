"""
文件存储（File Storage）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..utils.shm import CreateShm, EEGTYPE


class FileStorage:
    """文件存储类。"""
    
    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self.shm = CreateShm(master=False)
        self._current_file = None
    
    def connect_shm(self):
        """连接共享内存（兼容性方法）。"""
        pass
    
    def write_from_shm(self):
        """从共享内存读取数据并写入文件。"""
        try:
            import time
            
            # 检查是否开启保存
            if self.shm.getvalue('savedata') != 1:
                return False
            
            # 获取保存路径
            try:
                pth = self.shm.getPath()
            except:
                return False
            
            # 读取共享内存数据
            curdataindx = self.shm.getvalue('curdataindex')
            totalChsNum = (
                self.shm.getvalue('emgchs') +
                self.shm.getvalue('accchs') +
                self.shm.getvalue('glovechs')
            )
            
            if totalChsNum == 0 or curdataindx == 0:
                return False
            
            pp = int(curdataindx / totalChsNum)
            if pp == 0:
                return False
            
            # 读取数据
            dat = self.shm.eeg[:curdataindx].reshape(pp, totalChsNum).transpose()
            
            # 写入文件（.dat格式）
            file_path = Path(pth) if self.output_path is None else Path(self.output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果是新文件，写入头信息
            if self._current_file != file_path:
                if self._current_file:
                    try:
                        self._current_file.close()
                    except:
                        pass
                
                self._current_file = open(file_path, 'wb')
                
                # 写入头信息
                emgChs = self.shm.getvalue('emgchs')
                accChs = self.shm.getvalue('accchs')
                gloveChs = self.shm.getvalue('glovechs')
                srate = self.shm.getvalue('srate')
                
                typeLen = 8 if EEGTYPE == 'float64' else 4
                if typeLen == 4:
                    ay = np.array([7, 2, 2, srate, emgChs, accChs, gloveChs], dtype=np.int32)
                else:
                    ay = np.array([7, 2, 3, srate, emgChs, accChs, gloveChs], dtype=np.int32)
                
                self._current_file.write(ay.tobytes())
            
            # 写入数据
            self._current_file.write(dat.astype(EEGTYPE).tobytes())
            self._current_file.flush()
            
            return True
        except Exception as e:
            print(f"[FileStorage] ✗ 写入失败: {e}")
            return False
    
    def close(self):
        """关闭文件。"""
        if self._current_file:
            try:
                self._current_file.close()
            except:
                pass
            self._current_file = None












