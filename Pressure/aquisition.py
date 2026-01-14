import serial  # pip install pyserial
import serial.tools.list_ports
import os
import pandas as pd
from time import sleep
from pynput import keyboard
import ast


class KeyboardListener:
    def __init__(self, stop_key=keyboard.Key.esc):
        self._pressed = set()
        self._stop_key = stop_key
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._running = False

    def _on_press(self, key):
        try:
            char = key.char.lower() if key.char else None
            if char: self._pressed.add(char)
        except AttributeError:
            self._pressed.add(key)

    def _on_release(self, key):
        try:
            char = key.char.lower() if key.char else None
            if char: self._pressed.discard(char)
        except AttributeError:
            self._pressed.discard(key)
        
        if key == self._stop_key:
            self.stop()

    def start(self):
        if not self._running:
            self._listener.start()
            self._running = True

    def stop(self):
        if self._running:
            self._listener.stop()
            self._running = False

    def is_pressed(self, char: str) -> bool:
        return char.lower() in self._pressed

    @property
    def running(self):
        return self._running


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
            return ast.literal_eval(response)
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

files = [file for file in os.listdir('files') if '.csv' in file]
PATH  = f'files/test_{len(files)+1}.csv'
print(f"Arquivo de saída: {PATH}")

device = Device()
device.connect() 

print('Conectado! Pressione "q" para sair\n')
sleep(2)

data = []
kb = KeyboardListener()
kb.start()

while kb.running:
    if kb.is_pressed('q'):
        break

    values = device.get()
    
    try:
        values = ast.literal_eval(values)
    except:
        continue

    if values is not None:
        values['pressure'] *= 0.50
        data.append(values)
        print(f"Recebido: {values}")
    
    sleep(0.01)

kb.stop()
df = pd.DataFrame(data)
df.to_csv(PATH, index=False)
print(f"Dados salvos em {PATH}")