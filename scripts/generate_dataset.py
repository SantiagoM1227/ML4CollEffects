#import useful libraries
import numpy as np  # arrays and math
import matplotlib.pyplot as plt  # plots
import xtrack as xt  # tracking module of Xsuite
#%matplotlib widget

def plot_phase_space_with_profiles(x_array, px_array, title="",axis_labels = None, label=None, bins=120, difference=False)->None:
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(4, 1.3),
        height_ratios=(1.3, 4),
        hspace=0.05,
        wspace=0.05
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histpx = fig.add_subplot(gs[1, 1])

    # Scatter
    number_of_plots = len(x_array) if isinstance(x_array, list) else 1
    if number_of_plots > 1:
        for i in range(number_of_plots):
            x = x_array[i]
            px = px_array[i]
            lbl = label[i] if label is not None else None
            ax_scatter.scatter(x, px, s=2, alpha=0.4, label=lbl)
    else:
        x = x_array if not isinstance(x_array, list) else x_array[0]
        px = px_array if not isinstance(px_array, list) else px_array[0]
   
    ax_scatter.set_xlabel("x") if axis_labels is None else ax_scatter.set_xlabel(axis_labels[0])
    ax_scatter.set_ylabel("px") if axis_labels is None else ax_scatter.set_ylabel(axis_labels[1])
    if label is not None:
        ax_scatter.legend()

    # x profile
    if number_of_plots > 1:
        for i in range(number_of_plots):
            x = x_array[i]
            lbl = label[i] if label is not None else None
            ax_histx.hist(x, bins=bins, alpha=0.4, label=lbl)
            if difference and i == 1:  # If difference is True, plot the difference for the second dataset
                x_diff = x - (x_array[0] if not isinstance(x_array[0], list) else x_array[0][i])
                ax_histx.hist(x_diff, bins=bins, alpha=0.4, label=f"{lbl} Difference")
            
    else:        
        x = x_array if not isinstance(x_array, list) else x_array[0]
        ax_histx.hist(x, bins=bins, alpha=0.4)
    
    ax_histx.set_ylabel("count")
    ax_histx.tick_params(axis="x", labelbottom=False) 

    # px profile
    if number_of_plots > 1:
        for i in range(number_of_plots):
            px = px_array[i]
            lbl = label[i] if label is not None else None
            ax_histpx.hist(px, bins=bins, orientation="horizontal", alpha=0.4, label=lbl)
            if difference and i == 1:  # If difference is True, plot the difference for the second dataset
                px_diff = px - (px_array[0] if not isinstance(px_array[0], list) else px_array[0][i])
                ax_histpx.hist(px_diff, bins=bins, orientation="horizontal", alpha=0.4, label=f"{lbl} Difference")
    else:
        px = px_array if not isinstance(px_array, list) else px_array[0]
        ax_histpx.hist(px, bins=bins, orientation="horizontal", alpha=0.4)

    ax_histpx.set_xlabel("count")
    ax_histpx.tick_params(axis="y", labelleft=False)

    fig.suptitle(title)
    plt.show()


##################################

print('xtrack version', xt.__version__)

# Reference energy


part = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=1e9)  # Mass eV/c^2 Momentum in eV/c

# Element geometry
env=xt.Environment()
env.vars.default_to_zero = True
env['l_mq'] = 0.4


env.new('mq', xt.Quadrupole, length='l_mq')
env.new('qf1', 'mq', k1='kf1') # Focusing quadrupole  k1>0
env.new('qf2', 'mq', k1='kf2') # Focusing quadrupole  k1>0
env.new('qd1', 'mq', k1='kd1') # De-focusing quadrupole k1<0
#Markers
env.new("start_cell",xt.Marker)
env.new("end_cell",xt.Marker)


line = env.new_line(
    length=4, # 4 m length
    components=[
    env.place('start_cell', at=0),
    env.place('qf1', at=1),
    env.place('qd1', at=2),   # refer to the center by default
    env.place('qf2', at=3),
    env.place('end_cell',at=4),
    ]
)
line.particle_ref=part
line.build_tracker()   


kf1 = 1.0; kd1 = -1.0 ; kf2 = 1.0

env['kf1'] = kf1
env['kd1'] = kd1
env['kf2'] = kf2

rng = np.random.default_rng(56)
n_particles = 50000

p0 = line.build_particles(
    x     = rng.normal(0.0, 1e-3, n_particles),
    y     = rng.normal(0.0, 1e-3, n_particles),
    zeta  = rng.normal(0.0, 1e-3, n_particles),
    px    = rng.normal(0.0, 1e-3, n_particles),
    py    = rng.normal(0.0, 1e-3, n_particles),
    delta = rng.normal(0.0, 1e-4, n_particles)
)
line.track(p0, turn_by_turn_monitor="ONE_TURN_EBE")





def sample_initial_conditions(n_particles: int, rng: np.random.Generator) -> np.ndarray:
    x     = rng.normal(0.0, 1e-3, n_particles)
    y     = rng.normal(0.0, 1e-3, n_particles)
    zeta  = rng.normal(0.0, 1e-3, n_particles)
    px    = rng.normal(0.0, 1e-3, n_particles)
    py    = rng.normal(0.0, 1e-3, n_particles)
    delta = rng.normal(0.0, 1e-4, n_particles)

    ## it must be separated as {[x0,y0,zeta0,px0,py0,delta0], [x1,y1,zeta1,px1,py1,delta1], ...} -> column stack

    return np.column_stack([x, y, zeta, px, py, delta]).astype(np.float64)

def particles_to_6d(particles: xt.Particles) -> np.ndarray:
    # Henon order: [q, p] = [x, y, zeta, px, py, delta]
    return np.column_stack([
        np.array(particles.x),
        np.array(particles.y),
        np.array(particles.zeta),
        np.array(particles.px),
        np.array(particles.py),
        np.array(particles.delta),
    ]).astype(np.float64)


def data_builder(line: xt.Line, 
                 env,
                 total_particles=50000,
                 seed=42,
                 kf1_range=(0.2, 1.5),
                 kd1_range=(-2.5, -0.2),
                 kf2_range=(0.2, 1.5),
                 quad_iterations=1,
                 save_params=True) -> dict:
    
    rng = np.random.default_rng(seed)
    X_all, Y_all = [], []
    sampled_params = []

    for _ in range(quad_iterations):
        kf1 = rng.uniform(*kf1_range)
        kd1 = rng.uniform(*kd1_range)
        kf2 = rng.uniform(*kf2_range)

        env['kf1'] = kf1
        env['kd1'] = kd1
        env['kf2'] = kf2

        z0 = sample_initial_conditions(total_particles, rng)

        p = line.build_particles(
            x=z0[:, 0],
            y=z0[:, 1],
            zeta=z0[:, 2],
            px=z0[:, 3],
            py=z0[:, 4],
            delta=z0[:, 5],
        )

        line.track(p)

        z1 = particles_to_6d(p)

        mu = np.tile(np.array([kf1, kd1, kf2], dtype=np.float64), (total_particles, 1))
        X_batch = np.hstack([z0, mu]) if save_params else z0
        Y_batch = z1

        X_all.append(X_batch)
        Y_all.append(Y_batch)
        sampled_params.append((kf1, kd1, kf2))

    X = np.vstack(X_all)
    Y = np.vstack(Y_all)

    perm = rng.permutation(X.shape[0])   # fixed
    X = X[perm]
    Y = Y[perm]

    np.savez("xsuite_dataset.npz", X=X, Y=Y)
    print("Dataset shapes: X =", X.shape, ", Y =", Y.shape)
    print("Dataset saved to xsuite_dataset.npz")

    return {"X": X, "Y": Y, "params": sampled_params}

def build_operator_dataset(
    line,
    env,
    n_samples=2000,
    particles_per_sample=4000,
    seed=42,
    kf1_range=(0.1, 2.0),
    kd1_range=(-4.5, -0.05),
    kf2_range=(0.1, 2.0),
):
    rng = np.random.default_rng(seed)

    X_samples = []
    Y_samples = []
    MU_samples = []

    for _ in range(n_samples):
        kf1 = rng.uniform(*kf1_range)
        kd1 = rng.uniform(*kd1_range)
        kf2 = rng.uniform(*kf2_range)

        env["kf1"] = kf1
        env["kd1"] = kd1
        env["kf2"] = kf2

        z0 = sample_initial_conditions(particles_per_sample, rng)

        p = line.build_particles(
            x=z0[:, 0],
            y=z0[:, 1],
            zeta=z0[:, 2],
            px=z0[:, 3],
            py=z0[:, 4],
            delta=z0[:, 5],
        )
        line.track(p)
        z1 = particles_to_6d(p)

        X_samples.append(z0.astype(np.float32))          # [Np, 6]
        Y_samples.append(z1.astype(np.float32))          # [Np, 6]
        MU_samples.append(np.array([kf1, kd1, kf2], dtype=np.float32))

    X_samples = np.stack(X_samples, axis=0)   # [Ns, Np, 6]
    Y_samples = np.stack(Y_samples, axis=0)   # [Ns, Np, 6]
    MU_samples = np.stack(MU_samples, axis=0) # [Ns, 3]

    np.savez("xsuite_operator_dataset.npz", X=X_samples, Y=Y_samples, MU=MU_samples)
    print("Saved:", X_samples.shape, Y_samples.shape, MU_samples.shape)
    
    
# in generate_dataset.py
if __name__ == "__main__":
    build_operator_dataset(
        line,
        env,
        n_samples=200,
        particles_per_sample=4000,
        seed=42,
        kf1_range=(0.1, 2.0),
        kd1_range=(-4.5, -0.05),
        kf2_range=(0.1, 2.0),
    )

data = data_builder(line, 
             env, 
             total_particles=50000, 
             seed=42, 
             kf1_range=(0.1, 2), 
             kd1_range=(-4.5, -0.05), 
             kf2_range=(0.1, 2), 
             quad_iterations=1,
             save_params=False)


z_in_raw = data["X"]
z_out_raw = data["Y"]


x_data_generator = z_out_raw[:, 0]
px_data_generator = z_out_raw[:, 3]
y_data_generator = z_out_raw[:, 1]
py_data_generator = z_out_raw[:, 4]
zeta_data_generator = z_out_raw[:, 2]
delta_data_generator = z_out_raw[:, 5]

x = np.array(p0.x)
px = np.array(p0.px)
y = np.array(p0.y)
py = np.array(p0.py)
zeta = np.array(p0.zeta)
delta = np.array(p0.delta)

x_array = [x, x_data_generator]
px_array = [px, px_data_generator]
y_array = [y, y_data_generator]
py_array = [py, py_data_generator]
zeta_array = [zeta, zeta_data_generator]
delta_array = [delta, delta_data_generator]

label = ["Original", "Data Generator"]
x_axis_array = [x_array, y_array, zeta_array]
px_axis_array = [px_array, py_array, delta_array]

axis_labels = [["x", "px"] , ["y", "py"], ["zeta", "delta"]]

axis_index = 1 # 0 for x-px, 1 for y-py, 2 for zeta-delta

for axis_index in range(3):
    plot_phase_space_with_profiles(x_axis_array[axis_index], px_axis_array[axis_index], title=f"Transport for kf1={env['kf1']}, kd1={env['kd1']}, kf2={env['kf2']}", axis_labels=axis_labels[axis_index], label=label, difference=False)
    
