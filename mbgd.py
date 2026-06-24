import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from typing import Any
from sklearn.datasets import load_diabetes

# --- data ---

data: Any = load_diabetes(as_frame=True)
X = np.array(data.frame.bmi)
Y_raw = data.frame.target
Y_mean, Y_std = Y_raw.mean(), Y_raw.std()
Y = np.array((Y_raw - Y_mean) / Y_std)

lr = 0.001
BATCH_SIZE = 16
MAX_ITER = int(2e5)

# --- spaced sampling dentro de cada batch ---

def spaced_batch(rng: np.random.Generator, n: int, batch_size: int) -> np.ndarray:
    """Muestreo sistemático con jitter: garantiza índices bien espaciados."""
    segment_size = n / batch_size
    offsets = rng.uniform(0, segment_size, size=batch_size)
    starts = np.arange(batch_size) * segment_size
    indices = (starts + offsets).astype(int)
    indices = np.clip(indices, 0, n - 1)
    return indices

# --- mini-batch gradient descent ---
params = np.array([-2.0, -1.0])
N = len(X)

errors_batch: list[float] = []   # MSE del batch (ruidoso, lo que guía el paso)
errors_full: list[float] = []    # MSE completo (para graficar sobre la superficie)
b0s: list[float] = []
w1s: list[float] = []
batch_indices_log: list[np.ndarray] = []  # para visualizar qué datos se usaron

rng = np.random.default_rng(42)
iter_count = 0

while iter_count < MAX_ITER:
    b0, w1 = params[0], params[1]

    # batch bien espaciado
    batch_idx = spaced_batch(rng, N, BATCH_SIZE)
    X_batch = X[batch_idx]
    Y_batch = Y[batch_idx]

    Y_pred_batch = b0 + w1 * X_batch
    residuals = Y_batch - Y_pred_batch

    # gradientes promediados sobre el batch
    grad_b0 = -2 * residuals.mean()
    grad_w1 = -2 * (X_batch * residuals).mean()

    # MSE del batch (ruidoso)
    mse_batch = (residuals ** 2).mean()

    # MSE completo en este punto (para superficie)
    Y_pred_full = b0 + w1 * X
    mse_full = ((Y - Y_pred_full) ** 2).mean()

    b0s.append(b0)
    w1s.append(w1)
    errors_batch.append(mse_batch)
    errors_full.append(mse_full)
    batch_indices_log.append(batch_idx)

    params[0] -= lr * grad_b0
    params[1] -= lr * grad_w1

    iter_count += 1

b0s = np.array(b0s)
w1s = np.array(w1s)
errors_batch = np.array(errors_batch)
errors_full = np.array(errors_full)
total_steps = len(b0s)

print(f"Mini-batch — iteraciones: {total_steps}")
print(f"b0 final: {b0s[-1]:.4f}, w1 final: {w1s[-1]:.4f}")
print(f"MSE final (completo): {errors_full[-1]:.4f}")

# --- batch GD (todos los datos) — mismos pasos y lr para comparación justa ---
params_bgd = np.array([-2.0, -1.0])
bgd_b0s: list[float] = []
bgd_w1s: list[float] = []
bgd_errors: list[float] = []

for _ in range(MAX_ITER):
    b0_bgd, w1_bgd = params_bgd
    Y_pred_bgd = b0_bgd + w1_bgd * X
    residuals_bgd = Y - Y_pred_bgd

    grad_b0_bgd = -2 * residuals_bgd.mean()
    grad_w1_bgd = -2 * (X * residuals_bgd).mean()
    mse_bgd = (residuals_bgd ** 2).mean()

    bgd_b0s.append(b0_bgd)
    bgd_w1s.append(w1_bgd)
    bgd_errors.append(mse_bgd)

    params_bgd[0] -= lr * grad_b0_bgd
    params_bgd[1] -= lr * grad_w1_bgd

bgd_b0s    = np.array(bgd_b0s)
bgd_w1s    = np.array(bgd_w1s)
bgd_errors = np.array(bgd_errors)

print(f"Batch GD   — b0: {bgd_b0s[-1]:.4f}, w1: {bgd_w1s[-1]:.4f}, MSE: {bgd_errors[-1]:.4f}")

# --- superficie MSE (calculada sobre todos los datos) ---
margin_b0 = max((b0s.max() - b0s.min()) * 0.25, 0.5)
margin_w1 = max((w1s.max() - w1s.min()) * 0.25, 0.5)

B0_vals = np.linspace(b0s.min() - margin_b0, b0s.max() + margin_b0, 30)
W1_vals = np.linspace(w1s.min() - margin_w1, w1s.max() + margin_w1, 30)

B0, W1 = np.meshgrid(B0_vals, W1_vals)

Y_pred_mesh = B0[:, :, np.newaxis] + W1[:, :, np.newaxis] * X  # (30, 30, N)
MSE_mesh = ((Y - Y_pred_mesh) ** 2).mean(axis=2)               # (30, 30)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 1: superficie 3D MSE + trayectoria
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 7))
fig.suptitle('Mini-Batch Gradient Descent — Superficie MSE', fontsize=12, fontweight='bold')
plt.subplots_adjust(bottom=0.22)

ax: Any = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(B0, W1, MSE_mesh, color='gray', linewidth=0.4, alpha=0.5)
ax.set_xlabel('b0')
ax.set_ylabel('w1')
ax.set_zlabel('MSE')

# trayectoria completa estática — dibujada una sola vez
ax.plot(b0s, w1s, errors_full, color='red', linewidth=1.5, alpha=0.7, label='trayectoria')

# solo el punto actual es dinámico — O(1) por frame
current_dot, = ax.plot([b0s[0]], [w1s[0]], [errors_full[0]], 'o',
                       color='yellow', markeredgecolor='red', markersize=9, zorder=5)

ax.legend(loc='upper right', fontsize=8)
state = {'step': 0}

def update(step: int):
    # solo mover el dot — sin reconstruir la trayectoria
    current_dot.set_data([b0s[step]], [w1s[step]])
    current_dot.set_3d_properties([errors_full[step]])

    ax.set_title(
        f'Paso {step + 1} / {total_steps}  |  '
        f'b0={b0s[step]:.4f}  w1={w1s[step]:.4f}  '
        f'MSE_full={errors_full[step]:.4f}  MSE_batch={errors_batch[step]:.4f}',
        fontsize=9
    )
    fig.canvas.draw_idle()

update(0)

# botones fig 1
ax_prev  = plt.axes([0.22, 0.10, 0.11, 0.05])
ax_next  = plt.axes([0.35, 0.10, 0.11, 0.05])
ax_reset = plt.axes([0.48, 0.10, 0.11, 0.05])
ax_end   = plt.axes([0.61, 0.10, 0.11, 0.05])

btn_prev  = widgets.Button(ax_prev,  '← Prev')
btn_next  = widgets.Button(ax_next,  'Next →')
btn_reset = widgets.Button(ax_reset, 'Reset')
btn_end   = widgets.Button(ax_end,   'Final →')

ax_slider = plt.axes([0.15, 0.04, 0.72, 0.03])
slider = widgets.Slider(ax_slider, 'Paso', 0, total_steps - 1, valinit=0, valstep=1)

def sync_slider(step):
    slider.set_val(step)

def on_next(event):
    if state['step'] < total_steps - 1:
        state['step'] += 1
        update(state['step'])
        sync_slider(state['step'])

def on_prev(event):
    if state['step'] > 0:
        state['step'] -= 1
        update(state['step'])
        sync_slider(state['step'])

def on_reset(event):
    state['step'] = 0
    update(0)
    sync_slider(0)

def on_end(event):
    state['step'] = total_steps - 1
    update(state['step'])
    sync_slider(state['step'])

def on_slider(val):
    state['step'] = int(slider.val)
    update(state['step'])

btn_next.on_clicked(on_next)
btn_prev.on_clicked(on_prev)
btn_reset.on_clicked(on_reset)
btn_end.on_clicked(on_end)
slider.on_changed(on_slider)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 2: scatter + recta + puntos del batch resaltados
# ──────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Mini-Batch GD — Datos y Convergencia MSE', fontsize=12, fontweight='bold')
plt.subplots_adjust(bottom=0.22, wspace=0.35)

ax2 = axes2[0]
ax_mse = axes2[1]

# scatter de todos los datos
ax2.scatter(X, Y, color='steelblue', alpha=0.35, s=15, label='todos los datos', zorder=2)
ax2.set_xlabel('bmi (normalizado)')
ax2.set_ylabel('target (normalizado)')
ax2.set_title('Regresión lineal')

x_line = np.linspace(X.min(), X.max(), 200)

# recta mini-batch (dinámica)
recta, = ax2.plot(x_line, b0s[0] + w1s[0] * x_line,
                  color='tomato', linewidth=2, zorder=4, label='Mini-Batch GD')

# recta batch GD en el mismo paso (dinámica)
recta_bgd, = ax2.plot(x_line, bgd_b0s[0] + bgd_w1s[0] * x_line,
                      color='limegreen', linewidth=2, linestyle='--', zorder=4, label='Batch GD')

# puntos del batch resaltados
scatter_batch = ax2.scatter([], [], color='orange', edgecolors='black',
                             s=60, zorder=5, label=f'mini-batch (n={BATCH_SIZE})', linewidths=0.8)
ax2.legend(fontsize=8)

# panel derecho: MSE a lo largo del tiempo
ax_mse.set_xlabel('Iteración')
ax_mse.set_ylabel('MSE')
ax_mse.set_title('Evolución del MSE')
ax_mse.set_xlim(0, total_steps)
ax_mse.set_ylim(0, max(errors_batch.max(), errors_full.max()) * 1.05)

# curvas completas estáticas — se dibujan una sola vez
ax_mse.plot(np.arange(total_steps), errors_batch,
            color='tomato', linewidth=0.8, alpha=0.3, linestyle=':', label='Mini-Batch GD (MSE batch)')
ax_mse.plot(np.arange(total_steps), bgd_errors,
            color='limegreen', linewidth=1.5, alpha=0.85, linestyle='--', label='Batch GD (MSE completo)')

# elementos dinamicos: solo vline + 2 dots — O(1) por frame, sin reconstruir arrays
vline        = ax_mse.axvline(x=0, color='white', linewidth=1, alpha=0.6)
mse_dot,     = ax_mse.plot([0], [errors_full[0]],  'o', color='yellow',
                            markeredgecolor='tomato',    markersize=8, zorder=5)
mse_bgd_dot, = ax_mse.plot([0], [bgd_errors[0]], 's', color='yellow',
                             markeredgecolor='limegreen', markersize=8, zorder=5)
ax_mse.legend(fontsize=7)

def update2(step: int, sync_slider: bool = True):
    # scatter: rectas (solo set_ydata, O(x_line) = O(200))
    recta.set_ydata(b0s[step] + w1s[step] * x_line)
    recta_bgd.set_ydata(bgd_b0s[step] + bgd_w1s[step] * x_line)

    # batch resaltado (16 puntos)
    bidx = batch_indices_log[step]
    scatter_batch.set_offsets(np.column_stack([X[bidx], Y[bidx]]))

    # MSE panel: solo mover vline y dots — O(1)
    vline.set_xdata([step, step])
    mse_dot.set_data([step], [errors_full[step]])
    mse_bgd_dot.set_data([step], [bgd_errors[step]])

    fig2.suptitle(
        f'Paso {step + 1} / {total_steps}  |  '
        f'Mini-Batch: b0={b0s[step]:.4f} w1={w1s[step]:.4f} MSE={errors_full[step]:.4f}  |  '
        f'Batch GD: b0={bgd_b0s[step]:.4f} w1={bgd_w1s[step]:.4f} MSE={bgd_errors[step]:.4f}',
        fontsize=9
    )
    if sync_slider:
        slider2.set_val(step)
    fig2.canvas.draw_idle()

# botones fig 2 (en fig2)
ax_prev2  = fig2.add_axes([0.22, 0.10, 0.11, 0.05])
ax_next2  = fig2.add_axes([0.35, 0.10, 0.11, 0.05])
ax_reset2 = fig2.add_axes([0.48, 0.10, 0.11, 0.05])
ax_end2   = fig2.add_axes([0.61, 0.10, 0.11, 0.05])

btn_prev2  = widgets.Button(ax_prev2,  '← Prev')
btn_next2  = widgets.Button(ax_next2,  'Next →')
btn_reset2 = widgets.Button(ax_reset2, 'Reset')
btn_end2   = widgets.Button(ax_end2,   'Final →')

ax_slider2 = fig2.add_axes([0.15, 0.04, 0.72, 0.03])
slider2 = widgets.Slider(ax_slider2, 'Paso', 0, total_steps - 1, valinit=0, valstep=1)

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

def on_end2(event):
    state['step'] = total_steps - 1
    update2(state['step'])

def on_slider2(val):
    state['step'] = int(slider2.val)
    update2(state['step'])

btn_next2.on_clicked(on_next2)
btn_prev2.on_clicked(on_prev2)
btn_reset2.on_clicked(on_reset2)
btn_end2.on_clicked(on_end2)
slider2.on_changed(on_slider2)

update2(0)  # llamar después de que slider2 esté definido

plt.show()