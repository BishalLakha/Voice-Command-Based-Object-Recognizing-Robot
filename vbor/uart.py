import serial
import time

#port = serial.Serial("/dev/ttyAMA0", baudrate=115200, timeout=0)
port = serial.Serial()


def send(value):
    port.write(value)


def read():
    value = port.read()
    return value


def close():
    serial.close()
