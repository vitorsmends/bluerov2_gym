import os
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 0. CONFIGURAÇÃO DE SAÍDA
# ==========================================
OUTPUT_DIR = "plots_disturbance"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(name):
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")

    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


# ==========================================
# 1. GERADOR DE DISTÚRBIO
# ==========================================
class DisturbanceGenerator:
    """
    Gerador standalone do distúrbio de corrente baseado na mesma lógica
    do Dynamics._init_jonswap() e Dynamics._jonswap_current().
    """

    SEA_CONFIGS = {
        "calm": {
            "Hs": 1.0,
            "Tp": 6.0,
            "gamma": 3.3,
            "N": 64,
            "wave_dir": (0.5, 0.5),
            "scale": 0.5,
            "max_current": 0.7,
            "alpha_wave": 0.02,
            "noise_std": 0.01,
            "alpha_noise": 0.3,
            "seed": 42,
        },
        "storm": {
            "Hs": 5.5,
            "Tp": 11.0,
            "gamma": 4.5,
            "N": 64,
            "wave_dir": (0.7, 0.7),
            "scale": 2.0,
            "max_current": 1.2,
            "alpha_wave": 0.03,
            "noise_std": 0.05,
            "alpha_noise": 0.5,
            "seed": 42,
        },
    }

    def __init__(self, sea_state="calm", dt=0.1):
        if sea_state not in self.SEA_CONFIGS:
            raise ValueError("sea_state deve ser 'calm' ou 'storm'.")

        self.sea_state = sea_state
        self.dt = dt
        self.g = 9.81

        cfg = self.SEA_CONFIGS[sea_state]

        self.wave_dir = np.array(cfg["wave_dir"], dtype=float)
        norm = np.linalg.norm(self.wave_dir)
        if norm < 1e-9:
            self.wave_dir = np.array([1.0, 0.0], dtype=float)
        else:
            self.wave_dir /= norm

        self.scale = cfg["scale"]
        self.max_current = cfg["max_current"]
        self.alpha_wave = cfg["alpha_wave"]
        self.noise_std = cfg["noise_std"]
        self.alpha_noise = cfg["alpha_noise"]

        self._rng = np.random.default_rng(cfg["seed"])
        self._t = 0.0
        self._nu_c_filt = np.zeros(3, dtype=float)
        self._noise_filt = np.zeros(3, dtype=float)
        self._waves = []

        self._init_jonswap(
            Hs=cfg["Hs"],
            Tp=cfg["Tp"],
            gamma=cfg["gamma"],
            N=cfg["N"],
        )

    def _init_jonswap(self, Hs, Tp, gamma, N):
        wp = 2.0 * np.pi / Tp
        dw = 3.0 * wp / N
        alpha_js = 0.076 * Hs ** 2

        for i in range(1, N + 1):
            omega = i * dw
            sigma = 0.07 if omega <= wp else 0.09
            r = np.exp(-((omega - wp) ** 2) / (2.0 * sigma ** 2 * wp ** 2))
            S = (
                alpha_js
                * self.g ** 2
                * omega ** -5
                * np.exp(-1.25 * (wp / omega) ** 4)
                * gamma ** r
            )
            a = np.sqrt(2.0 * S * dw)
            phase = self._rng.uniform(0.0, 2.0 * np.pi)
            self._waves.append((omega, a, phase))

    def step(self):
        """
        Retorna nu_c = [u_c, v_c, 0] no instante atual.
        """
        self._t += self.dt
        u_wave_scalar = 0.0

        for omega, a, phase in self._waves:
            u_wave_scalar += a * omega * np.cos(omega * self._t + phase)

        nu_wave = np.array([
            u_wave_scalar * self.wave_dir[0],
            u_wave_scalar * self.wave_dir[1],
            0.0
        ], dtype=float)

        nu_wave *= self.scale

        mag = np.linalg.norm(nu_wave)
        if mag > self.max_current:
            nu_wave = (nu_wave / mag) * self.max_current

        noise = self.noise_std * self._rng.standard_normal(3)
        noise[2] = 0.0

        self._noise_filt = (
            (1.0 - self.alpha_noise) * self._noise_filt
            + self.alpha_noise * noise
        )

        nu_c_raw = nu_wave + self._noise_filt

        self._nu_c_filt = (
            (1.0 - self.alpha_wave) * self._nu_c_filt
            + self.alpha_wave * nu_c_raw
        )

        return self._nu_c_filt.copy()

    def simulate(self, total_time=120.0):
        n_steps = int(total_time / self.dt)
        time = np.arange(n_steps) * self.dt
        nu_hist = np.zeros((n_steps, 3), dtype=float)

        for i in range(n_steps):
            nu_hist[i] = self.step()

        return time, nu_hist


# ==========================================
# 2. RESUMO
# ==========================================
def print_summary(data, sea_state):
    magnitude = np.linalg.norm(data[:, :2], axis=1)

    print(f"\n=== RESUMO DO CENÁRIO: {sea_state.upper()} ===")
    print(f"u_c  -> min={data[:, 0].min():.4f}, max={data[:, 0].max():.4f}, std={data[:, 0].std():.4f}")
    print(f"v_c  -> min={data[:, 1].min():.4f}, max={data[:, 1].max():.4f}, std={data[:, 1].std():.4f}")
    print(f"|nu| -> min={magnitude.min():.4f}, max={magnitude.max():.4f}, std={magnitude.std():.4f}")


# ==========================================
# 3. PLOTS
# ==========================================
def plot_component_comparison(time, calm_data, storm_data, component_idx, ylabel, filename, config):
    plt.figure(figsize=(6.5, 4.2))

    plt.plot(
        time,
        calm_data[:, component_idx],
        label="Calm",
        color=config["Calm"]["color"],
        linestyle=config["Calm"]["ls"],
        linewidth=1.8,
    )

    plt.plot(
        time,
        storm_data[:, component_idx],
        label="Storm",
        color=config["Storm"]["color"],
        linestyle=config["Storm"]["ls"],
        linewidth=1.8,
    )

    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(filename)


def plot_magnitude_comparison(time, calm_data, storm_data, filename, config):
    calm_mag = np.linalg.norm(calm_data[:, :2], axis=1)
    storm_mag = np.linalg.norm(storm_data[:, :2], axis=1)

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(
        time,
        calm_mag,
        label="Calm",
        color=config["Calm"]["color"],
        linestyle=config["Calm"]["ls"],
        linewidth=1.8,
    )

    plt.plot(
        time,
        storm_mag,
        label="Storm",
        color=config["Storm"]["color"],
        linestyle=config["Storm"]["ls"],
        linewidth=1.8,
    )

    plt.xlabel("Time [s]")
    plt.ylabel(r"Disturbance Magnitude $||\nu_c||$ [m/s]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(filename)


def plot_separate_state(time, data, sea_state, filename, color_main):
    magnitude = np.linalg.norm(data[:, :2], axis=1)

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(
        time, data[:, 0],
        label=r"$u_c$",
        color=color_main,
        linestyle="-",
        linewidth=1.8,
    )

    plt.plot(
        time, data[:, 1],
        label=r"$v_c$",
        color=color_main,
        linestyle="--",
        linewidth=1.8,
    )

    plt.plot(
        time, magnitude,
        label=r"$||\nu_c||$",
        color=color_main,
        linestyle=":",
        linewidth=2.0,
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Current Velocity [m/s]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    save_plot(filename)


# ==========================================
# 4. EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
    })

    config = {
        "Calm": {"color": "#1f77b4", "ls": "-"},
        "Storm": {"color": "#d62728", "ls": "--"},
    }

    total_time = 120.0
    dt = 0.1

    gen_calm = DisturbanceGenerator(sea_state="calm", dt=dt)
    gen_storm = DisturbanceGenerator(sea_state="storm", dt=dt)

    time, calm_data = gen_calm.simulate(total_time=total_time)
    _, storm_data = gen_storm.simulate(total_time=total_time)

    print_summary(calm_data, "calm")
    print_summary(storm_data, "storm")

    # ==========================================
    # FIGURA 1: u_c COMPARAÇÃO
    # ==========================================
    plot_component_comparison(
        time=time,
        calm_data=calm_data,
        storm_data=storm_data,
        component_idx=0,
        ylabel=r"$u_c$ [m/s]",
        filename="fig_disturbance_uc_comparison",
        config=config,
    )

    # ==========================================
    # FIGURA 2: v_c COMPARAÇÃO
    # ==========================================
    plot_component_comparison(
        time=time,
        calm_data=calm_data,
        storm_data=storm_data,
        component_idx=1,
        ylabel=r"$v_c$ [m/s]",
        filename="fig_disturbance_vc_comparison",
        config=config,
    )

    # ==========================================
    # FIGURA 3: MAGNITUDE COMPARAÇÃO
    # ==========================================
    plot_magnitude_comparison(
        time=time,
        calm_data=calm_data,
        storm_data=storm_data,
        filename="fig_disturbance_magnitude_comparison",
        config=config,
    )

    # ==========================================
    # FIGURA 4: SÉRIE TEMPORAL - CALM
    # ==========================================
    plot_separate_state(
        time=time,
        data=calm_data,
        sea_state="calm",
        filename="fig_disturbance_timeseries_calm",
        color_main=config["Calm"]["color"],
    )

    # ==========================================
    # FIGURA 5: SÉRIE TEMPORAL - STORM
    # ==========================================
    plot_separate_state(
        time=time,
        data=storm_data,
        sea_state="storm",
        filename="fig_disturbance_timeseries_storm",
        color_main=config["Storm"]["color"],
    )

    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    main()