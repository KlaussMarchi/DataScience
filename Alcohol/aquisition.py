import serial  # pip install pyserial
import serial.tools.list_ports
import os
import pandas as pd
from time import sleep
import ast


class Device:
    def __init__(self):
        self.device = None

    def connect(self, port=None):
        if port is None:
            port = self.getPort()
        
        print(f"Conectando em {port}...")
        self.device = serial.Serial(port, 115200, timeout=0.1) 

    def get(self):
        if self.device.in_waiting == 0:
            return None

        try:
            line = self.device.readline().decode('utf-8')
            return None if not line else line.strip()
        except Exception as e:
            return None
        
    def getList(self):
        started = False
        endded  = False
        response = ''

        while not endded:
            data = self.get()

            if data is None:
                continue

            if not started and '[' in data:
                data    = data[data.find('['):]
                started = True

            if not endded and ']' in data:
                data = data[:data.find(']')+1]
                endded = True
            
            if started or endded:
                response += data

            if endded:
                break

        try:
            return [int(val) for val in ast.literal_eval(response)]
        except:
            return None
        

    def getPort(self):
        ports = serial.tools.list_ports.comports()
        target_port = None

        for port in ports:
            if 'usb' in port.description.lower():
                target_port = port.device
                break
        
        if target_port is None and len(ports) > 0:
            target_port = ports[0].device
            
        if target_port is None:
            raise Exception("Nenhuma porta serial encontrada!")

        return target_port
    
    def send(self, data):
        if self.device and self.device.is_open:
            self.device.write(f'{data}\r\n'.encode())


if not os.path.exists('files'):
    os.makedirs('files')


df = pd.read_csv('files/raw.csv')
print(df)

device = Device()
device.connect()

print('Conectado! Pressione "q" para sair.')
sleep(2)
data = []

while True:
    print('\niniciando teste')
    device.send('calibrate')
    sleep(3.0)

    values = device.getList()
    print(f"Recebido: {values}\n")

    if values is None:
        continue

    result = int(input('alcool: ').strip())
    info   = {'data': values, 'alcohol': bool(result)}
    data.append(info)

    if input('deseja continuar? ') == 'n':
        print('encerrando')
        break

data = pd.DataFrame(data)
df   = pd.concat([df, data], axis=0)
df.to_csv('files/raw.csv', index=False)