# Aplikacja renderujaca gre PONG z Arduino na komputer.
# Cala gra liczona jest na Arduino
# Ta aplikacja umozliwia wyswietlanie obrazu z Arduino

import sys, time
import pygame
import serial
import serial.tools.list_ports

# USTAWIENIE OKNA GRY
WIN_W, WIN_H = 800, 480
FPS = 120

# SKALOWANIE
# - skalowanie jest potrzebne gdyz na takich wymiarach byl trenowany model
W_SRC, H_SRC = 128, 64
SX = WIN_W / W_SRC
SY = WIN_H / H_SRC

# KOMUNIKACJA Z ARDUINO
def open_arduino_serial(baud=115200, timeout=0):
    ports = list(serial.tools.list_ports.comports())
    preferred = [p.device for p in ports if "Arduino" in (p.description or "") or "CH340" in (p.description or "")]
    candidates = preferred or [p.device for p in ports]
    for dev in candidates:
        try:
            ser = serial.Serial(dev, baudrate=baud, timeout=timeout)
            time.sleep(0.8)
            ser.reset_input_buffer()
            print(f"[OK] Połączono z {dev} @ {baud}")
            return ser
        except Exception:
            continue
    print("[ERR] Nie znaleziono Arduino. Podaj port ręcznie w kodzie (serial.Serial('COM5', 115200)).")
    return None

# ODBIOR STANU GRY
def read_latest_state(ser):
    # dzialanie na krotkach - dla bezpieczenstwa, nie chcemy modyfikowac zadnych parametrow
    # zwracana jest krotka 6 liczb albo none jezeli nic nie zostalo odczytane
    if ser is None:
        return None
    last = None
    try:
        while True:
            line = ser.readline()
            if not line:
                break
            last = line
    except Exception:
        return None
    if not last:
        return None
    try:
        s = last.decode("utf-8", errors="ignore").strip()
        if not s.startswith("S,"):
            return None
        _, bx, by, pay, ppy, sai, spl = s.split(",")
        return (int(bx), int(by), int(pay), int(ppy), int(sai), int(spl))
    except Exception:
        return None

# WYSYLANIE STEROWANIA
def send_controls(ser, up, down):
    if ser is None:
        return
    try:
        msg = f"C,{1 if up else 0},{1 if down else 0}\n"
        ser.write(msg.encode("utf-8"))
    except Exception:
        pass

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Gra PONG")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    ser = open_arduino_serial()

    # ostatni znany stan (żeby nie mrugało zanim przyjdą dane)
    bx=64; by=32; p_ai=24; p_pl=24; score_ai=0; score_pl=0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # wejście użytkownika (klawiatura)
        keys = pygame.key.get_pressed()
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]

        # wyślij sterowanie do Arduino
        send_controls(ser, up, down)

        # odbierz najnowszy stan
        st = read_latest_state(ser)
        if st:
            bx, by, p_ai, p_pl, score_ai, score_pl = st

        # rysowanie
        screen.fill((0,0,0))
        # linia środkowa
        for y in range(0, WIN_H, 18):
            pygame.draw.rect(screen, (255,255,255), (WIN_W//2 - 2, y, 4, 10))

        # paletki i piłka (ze skalowaniem)
        pygame.draw.rect(screen, (255,255,255), (int(4*SX), int(p_ai*SY), int(2*SX), int(16*SY)))                       # AI (lewa)
        pygame.draw.rect(screen, (255,255,255), (int((W_SRC-4-2)*SX), int(p_pl*SY), int(2*SX), int(16*SY)))             # Player (prawa)
        pygame.draw.rect(screen, (255,255,255), (int(bx*SX), int(by*SY), int(2*SX), int(2*SY)))                         # Ball

        # HUD
        screen.blit(font.render(f"AI:  {score_ai}", True, (255,255,255)), (20, 10))
        screen.blit(font.render(f"GRACZ: {score_pl}", True, (255,255,255)), (WIN_W - 140, 10))
        pygame.display.flip()

    if ser:
        try: ser.close()
        except: pass
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
