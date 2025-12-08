"""Robot communication interface"""
import socket
import time
import math

class RobotController:
    def __init__(self, ip, port, timeout=4):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.connect(timeout)
    
    def connect(self, timeout):
        """Connect to robot"""
        try:
            self.socket.connect((self.ip, self.port))
            time.sleep(timeout)
            print(f"Connected to robot at {self.ip}:{self.port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            raise
    
    def send_command(self, cmd):
        """Send command to robot"""
        if not cmd.endswith('\r\n'):
            cmd += '\r\n'
        self.socket.sendall(cmd.encode())
    
    def receive_data(self):
        """Receive data from robot"""
        data = self.socket.recv(1024)
        return data.decode()
    
    def initialize_odometry(self):
        """Initialize odometry to zero"""
        self.send_command('set "$odox" 0')
        self.send_command('set "$odoy" 0')
        self.send_command('set "$odoth" 0')
        print("Odometry initialized")
        time.sleep(1)
    
    def get_odometry(self):
        """Get current odometry (odox, odoy, odoth)"""
        flag = -1
        cmd = 'eval $odox ; $odoy ; $odoth'
        self.send_command(cmd)
        
        while flag < 0:
            data = self.receive_data()
            numbers = data.strip().split()
            
            if numbers[-1] != 'queued' and numbers[-1] != 'flushed':
                try:
                    odox, odoy, odoth = [float(num) for num in numbers[-3:]]
                    flag = 1
                except ValueError:
                    continue
        
        return odox, odoy, odoth
    
    def drive_to(self, odox, odoy, theta, velocity):
        """Send drive command to robot"""
        self.send_command('flushcmds')
        cmd = f"driveon {odox} {odoy} {round(theta, 2)} @v{velocity}"
        self.send_command(cmd)
    
    def drive_with_timeout(self, odox, odoy, theta, velocity, timeout=0.3):
        """Send drive command with timeout condition"""
        self.send_command('flushcmds')
        cmd = f"driveon {odox} {odoy} {round(theta, 2)} @v{velocity}:($cmdtime>{timeout})"
        self.send_command(cmd)
    
    def stop(self):
        """Stop the robot"""
        self.send_command('flushcmds')
        self.send_command('stop')
        time.sleep(0.1)
    
    def idle(self):
        """Set robot to idle"""
        self.send_command('idle')
    
    def disconnect(self):
        """Close connection"""
        self.send_command('flushcmds')
        self.send_command('idle')
        self.send_command('exit')
        self.socket.close()
        print("Disconnected from robot")
