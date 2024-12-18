import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Parameters
L = 1               # Length of the beam
N = 100             # Number of spatial divisions
x = np.linspace(0, L, N)  # Spatial coordinates
dx = x[1] - x[0]          # Spatial step size

# Mass values
m_values = [1, 10, 100]  # Different masses

# Position of the point mass
load_position = N // 2

def delta_func(N, position):
    delta = np.zeros(N)
    delta[position] = 1 / dx
    return delta

# Delta function for the point mass
delta = delta_func(N, load_position)

# Fourth derivative matrix (finite difference)
coeff = 1 / dx**4
diagonals = [
    np.full(N, 6) * coeff,       # Main diagonal
    np.full(N-1, -4) * coeff,    # First off-diagonals
    np.full(N-2, 1) * coeff      # Second off-diagonals
]
D4 = diags(diagonals, [0, 1, 2], shape=(N, N)).toarray()
D4 += D4.T  # Make the matrix symmetric

# Apply boundary conditions (fixed ends)
D4[0, :] = D4[-1, :] = 0
D4[:, 0] = D4[:, -1] = 0

# Initial conditions: triangular shape with w = -1 at x = 1/2
def initial_conditions():
    w0 = np.zeros(N)
    midpoint = L / 2
    for i in range(N):
        if x[i] <= midpoint:
            w0[i] = -1 * x[i] / midpoint
        else:
            w0[i] = -1 * (L - x[i]) / midpoint
    w1 = np.zeros(N)  # Initial velocity
    return np.concatenate([w0, w1])

# Modified equation of motion
def beam_vibration(t, y, m):
    w = y[:N]
    w_t = y[N:]
    dw_dt = w_t
    dw_tt = -D4 @ w - m * delta * w_t[load_position]
    dw_tt[0] = dw_tt[-1] = 0  # Boundary conditions: zero acceleration at the ends
    return np.concatenate([dw_dt, dw_tt])

# Time settings
t_span = (0, 0.001)
t_eval = np.linspace(0, 0.001, 100)

# Plot and animation setup
fig, ax = plt.subplots(figsize=(10, 6))

# 初期条件ラインのプロット（初期時刻の変位分布）
y0 = initial_conditions()
w0_initial = y0[:N]
initial_line, = ax.plot(x, w0_initial, 'k--', lw=1, label='Initial shape')

line, = ax.plot(x, np.zeros(N), 'b-', lw=2, label='Current displacement')
ax.set_xlim(0, L)
ax.set_ylim(-2.0, 2.0)
ax.set_xlabel('x')
ax.set_ylabel('Displacement w(x, t)')
ax.set_title('Beam Displacement Animation')
ax.legend()

time_template = 'Time = {:.2f} s, Mass = {}'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i, sol, m):
    line.set_ydata(sol.y[:N, i])
    time_text.set_text(time_template.format(sol.t[i], m))
    return line, time_text

for m in m_values:
    print(f"m={m}の計算をしてます！")
    y0 = initial_conditions()
    sol = solve_ivp(beam_vibration, t_span, y0, t_eval=t_eval, args=(m,), method='RK45')
    
    ani = FuncAnimation(fig, animate, frames=len(t_eval), fargs=(sol, m), interval=50, blit=True)
    
    # アニメーションを保存 (ffmpeg等が必要)
    filename = f'beam_animation_mass_{m}.mp4'
    ani.save(filename, fps=60, dpi=300, extra_args=['-vcodec', 'libx264'])
    
print("すべての計算が終了しました！")