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

# Agent training
The agent training environment is written in the file train_pong_dqn.py. Initially, you can manually change the neural network parameters and training parameters in the code, for example:
```
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

STATE_DIM = 5            # [x, y, vx, vy, py] -> state vector
N_ACTIONS = 3            # 0=UP, 1=STAY, 2=DOWN -> actions that can be performed by the agent (3)
HIDDEN = 16              # do test 8/32 -> neurons in the hidden layer
GAMMA = 0.99             # discount factor -> 0.99 for the test (the agent remembers almost the entire previous action)
LR = 1e-3                # pace of learning
BUFFER_SIZE = 50_000
BATCH_SIZE = 256
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 30_000 # the initial setting of 1 means that the agent acts randomly
TARGET_SYNC = 2000
MAX_STEPS = 100_000
PRINT_EVERY = 2000

CKPT_EVERY = 2000        # after how many steps save a permanent checkpoint
KEEP_CHECKPOINTS = 5     # how many last checkpoints to keep
```
The training script uses the Deep Q-Learning (DQN) algorithm.
The agent observes a 5-dimensional state vector [x, y, vx, vy, py], representing ball position, velocity and paddle position.
At each step, the model outputs one of three possible actions: move up, stay, or move down.

The replay buffer (BUFFER_SIZE, BATCH_SIZE) allows the agent to learn from past experiences, improving training stability.
The EPS_START, EPS_END, and EPS_DECAY parameters control the ε-greedy exploration strategy – the agent starts with random actions and gradually shifts toward exploitation as training progresses.

Every TARGET_SYNC steps, weights from the main (policy) network are copied to the target network, which helps stabilize Q-value estimation.

Model checkpoints are saved in the artifacts/checkpoints/ directory, and the latest trained model is automatically exported to:
- artifacts/pong_dqn.pt - PyTorch model file
- artifacts/pong_policy.h - C header with model weights (used on Arduino)
  
Training progress (average reward, loss, ε value) is printed to the console every PRINT_EVERY steps. You can adjust this to monitor the learning curve more frequently.
