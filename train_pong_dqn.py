# skrypt wykorzystujacy DQN (reinforcement learning) do nauki agenta do gry w grę PONG z uzytkownikiem
# trenowany agent operuje po lewej stronie ekranu

import math, random, os, textwrap, glob
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ZESTAW PARAMETROW DQN
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

STATE_DIM = 5            # [x, y, vx, vy, py] -> wektor stanu
N_ACTIONS = 3            # 0=UP, 1=STAY, 2=DOWN -> akcje mozliwe do wykonania przez agenta (3!!!!!)
HIDDEN = 16              # do test 8/32 -> neurony w warstwie ukrytej
GAMMA = 0.99             # wspolczynnik dyskontu -> 0.99 dla testu (agent pamieta prawie w calosci poprzednia akcje)
LR = 1e-3                # tempo nauki
BUFFER_SIZE = 50_000
BATCH_SIZE = 256
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 30_000 # poczatkowe ustawienie na 1 mowi o tym ze agent dziala losowo
TARGET_SYNC = 2000
MAX_STEPS = 100_000
PRINT_EVERY = 2000

# AUTOZAPIS
CKPT_EVERY = 2000        # co ile krokow zapisać trwaly checkpoint
KEEP_CHECKPOINTS = 5     # ile ostatnich checkpointow trzymac

# SYMULACJA PONGA - SRODOWISKO RL
@dataclass
class PongConfig:
    W: int = 128
    H: int = 64
    pX: int = 4
    pW: int = 2
    pH: int = 16
    pSpeed: float = 2.0
    bSize: int = 2
    v0x: float = 1.5
    v0y_range: Tuple[float, float] = (-1.0, 1.0)

class PongEnv:
    def __init__(self, cfg=PongConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        c = self.cfg
        self.bx, self.by = c.W/2, c.H/2
        self.bvx = c.v0x * (1 if random.random() < 0.5 else -1)
        self.bvy = random.uniform(*c.v0y_range)
        self.py = c.H/2 - c.pH/2
        self.t = 0
        return self._state()

    def step(self, action:int):
        c = self.cfg
        # akcja paletki
        if action == 0: self.py -= c.pSpeed
        elif action == 2: self.py += c.pSpeed
        self.py = max(0, min(self.py, c.H - c.pH))

        # ruch piłki
        self.bx += self.bvx; self.by += self.bvy
        # odbicia góra/dół
        if self.by <= 0 or self.by >= c.H - c.bSize:
            self.bvy = -self.bvy
            self.by = max(0, min(self.by, c.H - c.bSize))
        # odbicie od prawej ściany
        if self.bx >= c.W - c.bSize:
            self.bx = c.W - c.bSize; self.bvx = -abs(self.bvx)

        # kolizja z paletką (lewa)
        hit = False
        if self.bx <= c.pX + c.pW and (self.py <= self.by <= self.py + c.pH):
            self.bx = c.pX + c.pW + 1
            self.bvx = abs(self.bvx)
            offset = ((self.by - (self.py + c.pH/2)) / (c.pH/2))
            self.bvy += 0.6 * offset
            hit = True

        # nagrody
        reward = 0.0
        if hit: reward += 1.0
        #reward += -0.001 * abs((self.py + c.pH/2) - self.by)

        # porazka
        done = False
        if self.bx < 0:
            reward -= 2.0
            done = True

        self.t += 1
        if self.t >= 3000:
            done = True

        return self._state(), reward, done, {}

    def _state(self):
        # normalizacja zgodna z Arduino
        x = (self.bx - 64)/64.0
        y = (self.by - 32)/32.0
        vx = self.bvx/2.0
        vy = self.bvy/2.0
        py = (self.py - 32)/32.0
        return np.array([x,y,vx,vy,py], dtype=np.float32)

# SIEC DQN - MODEL
# proste MLP (Multilayer Perceptron), schemat: 5 -> 16 -> 3 (zdefiniowane na poczatku skryptu)
# xavier - wykorzystane do inicjalizacji wag (w celu optymalizacji wag na poczatku uczenia)
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, N_ACTIONS)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer: # bufor - nauka z losowo wybranych doswiadczen (dlatego zapis do krotki losowych doswiadczen)
    def __init__(self, cap:int):
        self.buf: Deque[Tuple[np.ndarray,int,float,np.ndarray,bool]] = deque(maxlen=cap)
    def push(self, s,a,r,s2,d): self.buf.append((s,a,r,s2,d))
    def sample(self, n:int):
        batch = random.sample(self.buf, n)
        s,a,r,s2,d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))
    def __len__(self): return len(self.buf)

# EKSPORT WTRENOWANEJ SIECI JAKO PLIKU NAGLOWKOWEGO W C
def tensor_to_c_array(t: torch.Tensor, name: str) -> str:
    arr = t.detach().cpu().numpy()
    flat = arr.reshape(-1)
    vals = ", ".join(f"{v:.8f}" for v in flat)
    wrapped = textwrap.fill(vals, width=100)
    return f"const float {name}[{flat.size}] = {{ {wrapped} }};"

def export_to_c_header(model: QNet, path="artifacts/pong_policy.h"):
    W1 = model.fc1.weight.detach().cpu().numpy()  # [H, 5]
    B1 = model.fc1.bias.detach().cpu().numpy()    # [H]
    W2 = model.fc2.weight.detach().cpu().numpy()  # [3, H]
    B2 = model.fc2.bias.detach().cpu().numpy()    # [3]

    def as_2d(name, mat):
        rows = []
        for r in mat:
            row = ", ".join(f"{v:.8f}" for v in r)
            rows.append("  {" + row + "}")
        body = ",\n".join(rows)
        return f"const float {name}[{{ROWS}}][{{COLS}}] = {{\n{body}\n}};".replace("{ROWS}", str(len(mat))).replace("{COLS}", str(len(mat[0])))

    header = f"""#pragma once
// 
#define NN_IN 5
#define NN_H {HIDDEN}
#define NN_OUT 3

{as_2d("NN_W1", W1)}
const float NN_B1[NN_H] = {{ {", ".join(f"{v:.8f}" for v in B1)} }};
{as_2d("NN_W2", W2)}
const float NN_B2[NN_OUT] = {{ {", ".join(f"{v:.8f}" for v in B2)} }};
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)

# ZAPIS CO KROK + CHECKPOINTY
def _safe_save(model: QNet, pt_path: str, h_path: str):
    os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), pt_path)
    export_to_c_header(model, path=h_path)

def _rotate_checkpoints(ckpt_dir: str, keep: int):
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "pong_dqn_*.pt")))
    excess = len(pts) - keep
    for p in pts[:max(0, excess)]:
        try:
            os.remove(p)
            h = p.replace("pong_dqn_", "pong_policy_").replace(".pt", ".h")
            if os.path.exists(h): os.remove(h)
        except OSError:
            pass

def maybe_save(model: QNet, step: int):
    # 1) “latest” – nadpisywane co krok
    _safe_save(model, "artifacts/pong_dqn_latest.pt", "artifacts/pong_policy_latest.h")

    # 2) trwały checkpoint co CKPT_EVERY
    if step % CKPT_EVERY == 0:
        ckpt_dir = "artifacts/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        pt_path = os.path.join(ckpt_dir, f"pong_dqn_{step:07d}.pt")
        h_path  = os.path.join(ckpt_dir, f"pong_policy_{step:07d}.h")
        _safe_save(model, pt_path, h_path)
        _rotate_checkpoints(ckpt_dir, KEEP_CHECKPOINTS)

# NARZEDZIA DQN
def select_action(q:QNet, state:np.ndarray, steps_done:int):
    # wybor akcji:
    # - liczenie biezacego epsilon
    # - losowanie
    # - eksploatacja
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    if random.random() < eps:
        return random.randrange(N_ACTIONS), eps
    with torch.no_grad():
        s = torch.from_numpy(state).unsqueeze(0)
        qvals = q(s)
        return int(torch.argmax(qvals, dim=1).item()), eps

def train_step(q, q_targ, optim_, buf:ReplayBuffer, device):
    # pojedynczy krok uczenia
    s,a,r,s2,d = buf.sample(BATCH_SIZE)
    s = torch.tensor(s, device=device)
    a = torch.tensor(a, device=device, dtype=torch.int64)
    r = torch.tensor(r, device=device)
    s2= torch.tensor(s2, device=device)
    d = torch.tensor(d, device=device)

    qvals = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target = r + GAMMA * (1.0 - d) * q_targ(s2).max(1)[0]
    loss = nn.functional.smooth_l1_loss(qvals, target)
    optim_.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), 1.0)
    optim_.step()
    return float(loss.item())

# TRENING
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PongEnv()
    q = QNet().to(device)
    q_targ = QNet().to(device)
    q_targ.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=LR)
    buf = ReplayBuffer(BUFFER_SIZE)

    s = env.reset()
    ep_ret, ep_len = 0.0, 0
    losses = []

    try:
        for step in range(1, MAX_STEPS+1):
            a, eps = select_action(q, s, step)
            s2, r, done, _ = env.step(a)
            buf.push(s,a,r,s2,done)
            s = s2
            ep_ret += r; ep_len += 1

            if len(buf) >= BATCH_SIZE:
                loss = train_step(q, q_targ, optimizer, buf, device)
                losses.append(loss)

            # zapis co krok (latest)
            maybe_save(q, step)

            if done:
                s = env.reset()
                if step % PRINT_EVERY < 10:
                    print(f"[step {step}] ep_ret={ep_ret:.2f} ep_len={ep_len} eps={eps:.2f}")
                ep_ret, ep_len = 0.0, 0

            if step % TARGET_SYNC == 0:
                q_targ.load_state_dict(q.state_dict())

            if step % PRINT_EVERY == 0 and losses:
                print(f"  avg_loss={np.mean(losses[-500:]):.4f} buffer={len(buf)}")

    except KeyboardInterrupt:
        print("\n  Trening przerwany ręcznie – zapisuję bieżący stan do artifacts/ ...")
        _safe_save(q, "artifacts/pong_dqn_latest.pt", "artifacts/pong_policy_latest.h")

    finally:
        # końcowy zapis “pewniaka”
        os.makedirs("artifacts", exist_ok=True)
        torch.save(q.state_dict(), "artifacts/pong_dqn.pt")
        export_to_c_header(q, path="artifacts/pong_policy.h")
        print("Zapisano końcowe pliki: artifacts/pong_dqn.pt oraz artifacts/pong_policy.h")

if __name__ == "__main__":
    train()
