# pong_with_ai_arduino
his repository contains a Pong game environment that communicates with Arduino UNO over serial port. It includes a Python visualization (Pygame), a reinforcement learning agent trained using DQN, and Arduino code that can reproduce AI behavior or send live game data.

# Concept
The project focuses on controlling a simple Pong game via serial communication with Arduino UNO. An agent trained with Deep Q-Learning (DQN) interacts with the game, while the Arduino can visualize or control the game state using analog input (potentiometer).

### Sample frame from the game:
![pong_1](https://github.com/user-attachments/assets/c80f015e-bb9d-44c7-95da-977c61eec1be)

# Before use:
This project uses Pycharm version 2024.1.3. You can install libraries in Pycharm Terminal.
- install ultralytics:
```
pip install ultralytics
```
- upgrade pip:
```
python.exe -m pip install --upgrade pip
```
- install opencv-python matplotlib:
```
 pip install ultralytics opencv-python matplotlib  
```
### Required hardware:
The game engine was written on Arduino Uno R3 which is connected to PC. Control is performed using a B10K potentiometer:

![schemat](https://github.com/user-attachments/assets/d00c0529-898a-42dc-9897-01b8086bc251)


