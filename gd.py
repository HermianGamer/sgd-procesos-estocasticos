import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from typing import Any
from sklearn.datasets import load_diabetes

# --- data ---

data: Any = load_diabetes(as_frame=True)
X = np.array(data.frame.bmi)
Y_raw = data.frame.target        # esto es lo que explota
Y_mean, Y_std = Y_raw.mean(), Y_raw.std()
Y = np.array((Y_raw - Y_mean) / Y_std)
lr = 0.0001

# --- gradient descent: calcula todos los pasos primero ---
params = np.array([-2.0, -1.0])
slopes = np.array([10.0, 10.0])

errors: list[float] = []
b0s: list[float] = []
w1s: list[float] = []

MAX_ITER = 40_000
iter_count = 0

while iter_count < MAX_ITER and (abs(lr*slopes[0])>=1e-400 and abs(lr*slopes[1])>=1e-400):
    b0, w1 = params[0], params[1]

    Y_pred = b0 + w1 * X                       # vectorizado
    residuals = Y - Y_pred

    slopes[0] = -2 * residuals.sum()
    slopes[1] = -2 * (X * residuals).sum()

    ssr = (residuals ** 2).sum()

    b0s.append(b0)
    w1s.append(w1)
    errors.append(ssr)

    params -= lr * slopes

    iter_count += 1

b0s = np.array(b0s)
w1s = np.array(w1s)
errors = np.array(errors)
total_steps = len(b0s)

# --- superficie SSR ---
margin_b0 = (b0s.max() - b0s.min()) * 0.2
margin_w1 = (w1s.max() - w1s.min()) * 0.2

B0_vals = np.linspace(b0s.min() - margin_b0, b0s.max() + margin_b0, 25)
W1_vals = np.linspace(w1s.min() - margin_w1, w1s.max() + margin_w1, 25)

B0, W1 = np.meshgrid(B0_vals, W1_vals)


Y_pred_mesh = B0[:, :, np.newaxis] + W1[:, :, np.newaxis] * X  # (25, 25, 3)
SSR_mesh = ((Y - Y_pred_mesh) ** 2).sum(axis=2)                 # (25, 25)

# --- figura ---
fig = plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.22)

ax: Any = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(B0, W1, SSR_mesh, color='gray', linewidth=0.5, alpha=0.6)
ax.set_xlabel('b0')
ax.set_ylabel('w1')
ax.set_zlabel('SSR')

# linea completa de fondo (opaca)
ax.plot(b0s, w1s, errors, color='red', linewidth=1, alpha=0.15)

# elementos dinamicos
path_line, = ax.plot([], [], [], color='red', linewidth=2)
current_dot, = ax.plot([], [], [], 'o',
                       color='yellow',
                       markeredgecolor='red',
                       markersize=8,
                       zorder=5)

title = ax.set_title('Paso 0 / {} — presiona Next'.format(total_steps))

# --- estado ---
state = {'step': 0}

def update(step: int):
    idx = step + 1
    path_line.set_data(b0s[:idx], w1s[:idx])
    path_line.set_3d_properties(errors[:idx])

    current_dot.set_data([b0s[step]], [w1s[step]])
    current_dot.set_3d_properties([errors[step]])

    ax.set_title(
        f'Paso {step + 1} / {total_steps}  |  '
        f'b0={b0s[step]:.4f}  w1={w1s[step]:.4f}  SSR={errors[step]:.4f}'
    )
    fig.canvas.draw_idle()

update(0)

# --- botones ---
ax_prev = plt.axes([0.25, 0.08, 0.12, 0.055])
ax_next = plt.axes([0.40, 0.08, 0.12, 0.055])
ax_reset = plt.axes([0.55, 0.08, 0.12, 0.055])

btn_prev  = widgets.Button(ax_prev,  '← Prev')
btn_next  = widgets.Button(ax_next,  'Next →')
btn_reset = widgets.Button(ax_reset, 'Reset')

def on_next(event):
    if state['step'] < total_steps - 1:
        state['step'] += 1
        update(state['step'])

def on_prev(event):
    if state['step'] > 0:
        state['step'] -= 1
        update(state['step'])

def on_reset(event):
    state['step'] = 0
    update(0)

btn_next.on_clicked(on_next)
btn_prev.on_clicked(on_prev)
btn_reset.on_clicked(on_reset)

# slider
ax_slider = plt.axes([0.15, 0.03, 0.7, 0.03])
slider = widgets.Slider(ax_slider, 'Paso', 0, total_steps - 1,
                        valinit=0, valstep=1)

def on_slider(val):
    state['step'] = int(slider.val)
    update(state['step'])

slider.on_changed(on_slider)




# --- segunda figura: scatter + recta ---
fig2, ax2 = plt.subplots()
plt.subplots_adjust(bottom=0.22)

ax2.scatter(X, Y, color='steelblue', alpha=0.5, s=20, label='datos')
ax2.set_xlabel('bmi (normalizado)')
ax2.set_ylabel('target (normalizado)')

x_line = np.linspace(X.min(), X.max(), 100)
recta, = ax2.plot(x_line, b0s[0] + w1s[0] * x_line, color='red', linewidth=2)
titulo2 = ax2.set_title('Paso 1 / {}'.format(total_steps))

# --- botones figura 2 ---
ax_prev2  = plt.axes([0.25, 0.08, 0.12, 0.055])
ax_next2  = plt.axes([0.40, 0.08, 0.12, 0.055])
ax_reset2 = plt.axes([0.55, 0.08, 0.12, 0.055])

btn_prev2  = widgets.Button(ax_prev2,  '← Prev')
btn_next2  = widgets.Button(ax_next2,  'Next →')
btn_reset2 = widgets.Button(ax_reset2, 'Reset')

ax_slider2 = plt.axes([0.15, 0.03, 0.7, 0.03])
slider2 = widgets.Slider(ax_slider2, 'Paso', 0, total_steps - 1,
                         valinit=0, valstep=1)

def update2(step: int):
    recta.set_ydata(b0s[step] + w1s[step] * x_line)
    ax2.set_title(
        f'Paso {step + 1} / {total_steps}  |  '
        f'b0={b0s[step]:.4f}  w1={w1s[step]:.4f}  SSR={errors[step]:.4f}'
    )
    slider2.set_val(step)
    fig2.canvas.draw_idle()

def on_next2(event):
    if state['step'] < total_steps - 1:
        state['step'] += 1
        update2(state['step'])

def on_prev2(event):
    if state['step'] > 0:
        state['step'] -= 1
        update2(state['step'])

def on_reset2(event):
    state['step'] = 0
    update2(0)

def on_slider2(val):
    state['step'] = int(slider2.val)
    update2(state['step'])

btn_next2.on_clicked(on_next2)
btn_prev2.on_clicked(on_prev2)
btn_reset2.on_clicked(on_reset2)
slider2.on_changed(on_slider2)



plt.show()

