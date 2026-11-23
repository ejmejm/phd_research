"""Asteroids RAM env, RAM decoding and the hard-coded policy from 01_env_and_policy.ipynb, plus data collection."""
import numpy as np
from ale_py import ALEInterface, LoggerMode, roms

ALEInterface.setLoggerMode(LoggerMode.Error)
FRAMESKIP = 4


class AsteroidsRAM:
    def __init__(self, mode=0, difficulty=0, seed=0):
        self.ale = ALEInterface()
        self.ale.setInt('random_seed', seed)
        self.ale.setFloat('repeat_action_probability', 0.0)
        self.ale.loadROM(roms.Asteroids)
        self.ale.setMode(mode)
        self.ale.setDifficulty(difficulty)
        self.actions = self.ale.getMinimalActionSet()

    def reset(self):
        self.ale.reset_game()
        self.prev_screen = self.ale.getScreenRGB()
        return self.ale.getRAM().copy()

    def step(self, a):
        reward = 0
        for k in range(FRAMESKIP):
            if k == FRAMESKIP - 1:
                self.prev_screen = self.ale.getScreenRGB()
            reward += self.ale.act(self.actions[a])
        return self.ale.getRAM().copy(), reward, self.ale.game_over()

    def render(self):
        # Asteroids and the ship are drawn on alternate frames; the max over the last two shows both.
        return np.maximum(self.prev_screen, self.ale.getScreenRGB())


# RAM layout (checked against the screen). Asteroids are in two lists, y at 3.. and 12.., each ending at y == 224, with
# x at +18 and size/colour at +36. Ship x, y at 73, 74 (y == 224 while hidden); heading in the low nibble of 60.
SIZES = [(16, 28), (8, 14), (4, 8)]  # large, medium, small (w, h) in pixels
WRAP = np.array([160, 178])  # the screen wraps every 160 px across and 178 px down


def x_pixel(v):
    # Atari coarse/fine horizontal position byte -> pixel column (decoding from OC_Atari).
    if v <= 1:
        return 64 if v == 0 else 4
    lo, add, sub = v & 15, 8 * (v >> 7), (v >> 4) & 7
    base = 97 + 15 * (lo // 2 - 1) if lo % 2 == 0 else 10 + 15 * ((lo - 1) // 2 - 1)
    return base + add - sub


def signed(v):
    return v - 256 if v > 127 else v


def decode(ram):
    # Ship centre (None while hidden), heading 0-15 (0 = up, counter-clockwise), asteroid centres (n, 2), in pixels.
    r = ram.astype(int)
    asteroids = []
    for start in (3, 12):
        for i in range(start, start + 8):
            if r[i] == 224:
                break
            c = r[i + 36] & 127
            w, h = SIZES[0 if c < 32 else 1 if c < 48 else 2]
            asteroids.append((x_pixel(r[i + 18]) + w / 2, 24 + 2 * signed(r[i]) + h / 2))
    ship = None if r[74] == 224 else np.array([x_pixel(r[73]) + 2.5, 23 + 2 * signed(r[74])])
    return ship, r[60] & 15, np.array(asteroids).reshape(-1, 2)


ACTIONS = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE',
           'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE']  # the minimal action set, in order
TEMPERATURE = 10  # px; the target is sampled with probability ∝ exp(-distance / TEMPERATURE)
SAFE = 60  # px; thrust only when no asteroid is closer than this
THRUST_PROB = 0.1


def policy(ram, t, rng):
    ship, heading, asteroids = decode(ram)
    if ship is None or len(asteroids) == 0:
        return ACTIONS.index('NOOP')
    d = (asteroids - ship + WRAP / 2) % WRAP - WRAP / 2
    dist = np.hypot(*d.T)
    p = np.exp(-(dist - dist.min()) / TEMPERATURE)
    dx, dy = d[rng.choice(len(d), p=p / p.sum())]
    # The 16 headings are evenly spaced when y is measured in RAM units (2 px).
    turn = (round(np.arctan2(-dx, -dy / 2) / (np.pi / 8)) - heading) % 16
    thrust = dist.min() > SAFE and rng.random() < THRUST_PROB
    name = ('UP' if thrust else '') + ('' if turn == 0 else 'LEFT' if turn <= 8 else 'RIGHT') + ('FIRE' if t % 2 == 0 else '')
    return ACTIONS.index(name or 'NOOP')


def collect(n_steps, mode=0, difficulty=0, seed=0):
    # Plays n_steps with the policy. ram is the stream of states; transition t goes ram[t] -> ram[t + 1] under action[t],
    # and valid[t] is False where ram[t + 1] is a reset after game over.
    env, rng = AsteroidsRAM(mode, difficulty, seed), np.random.default_rng(seed)
    ram, action, valid, t = [env.reset()], [], [], 0
    for _ in range(n_steps):
        a = policy(ram[-1], t, rng)
        next_ram, _, done = env.step(a)
        ram.append(next_ram); action.append(a); valid.append(True); t += 1
        if done:
            ram.append(env.reset()); action.append(0); valid.append(False); t = 0
    return dict(ram=np.array(ram), action=np.array(action, np.uint8), valid=np.array(valid))
