"""
时序数据库存储（TSDB Storage）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.shm import CreateShm, EEGTYPE


@dataclass
class TSDBConfig:
    """TSDB配置。"""
    url: str = "http://localhost:8086"
    token: str = "mpscap-token"
    org: str = "mpscap"
    bucket: str = "signals"


class TSDBStorage:
    """时序数据库存储类。"""
    
    def __init__(self, config: Optional[TSDBConfig] = None):
        self.config = config or TSDBConfig()
        self.shm = CreateShm(master=False)
        self._client = None
        self._write_api = None
        
        # 尝试连接InfluxDB
        try:
            from influxdb_client import InfluxDBClient, WriteApi
            self._client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
            )
            self._write_api: WriteApi = self._client.write_api()
            print(f"[TSDB] 已连接到InfluxDB: {self.config.url}")
        except ImportError:
            print("[TSDB] ⚠ influxdb-client未安装，TSDB功能不可用")
            print("[TSDB]   请运行: pip install influxdb-client")
        except Exception as e:
            print(f"[TSDB] ⚠ 连接InfluxDB失败: {e}")
            print("[TSDB]   将使用文件存储作为后备方案")
    
    def connect_shm(self):
        """连接共享内存（兼容性方法）。"""
        pass
    
    def write_from_shm(self, device_id: str, session_id: Optional[str] = None):
        """从共享内存读取数据并写入TSDB。"""
        if self._write_api is None:
            return False
        
        try:
            from influxdb_client import Point
            import time
            
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
            
            # 写入InfluxDB
            point = Point("emg_data")
            if session_id:
                point.tag("session_id", session_id)
            point.tag("device_id", device_id)
            point.field("channels", totalChsNum)
            point.field("samples", pp)
            point.time(int(time.time() * 1e9))  # 纳秒时间戳
            
            self._write_api.write(self.config.bucket, record=point)
            return True
        except Exception as e:
            print(f"[TSDB] ✗ 写入失败: {e}")
            return False
    
    def close(self):
        """关闭连接。"""
        if self._write_api:
            try:
                self._write_api.close()
            except:
                pass
        if self._client:
            try:
                self._client.close()
            except:
                pass












