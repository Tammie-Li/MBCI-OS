import paho.mqtt.client as mqtt
import time

def on_connect(client,userdata,flags,rc,properties):
    print(rc)
    if rc == 0:
        print('yes')

if __name__ == '__main__':
    timestampStr = str(int(time.time() * 100))
    nodename = 'BP' + '-' + timestampStr
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,nodename)
    client.on_connect = on_connect
    client.connect("127.0.0.1",1883,60)
    client.loop_forever()