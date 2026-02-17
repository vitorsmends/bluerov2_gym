from importlib import resources

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Certifique-se de que estes imports apontam para os arquivos onde 
# salvou as classes Dynamics e Reward atualizadas anteriormente
from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer


class BlueRov(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Carregamento do modelo 3D (Mantido original)
        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            self.model_path = str(asset_path)

        self.renderer = BlueRovRenderer()
        self.reward_fn = Reward()
        self.dynamics = Dynamics()
        
        # -----------------------------------------------------------
        # 1. Definição do Estado Inicial (12 variáveis para 6-DoF)
        # -----------------------------------------------------------
        self.state = {
            # Posição e Orientação (NED - North East Down)
            "x": 0.0, "y": 0.0, "z": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            # Velocidades no Referencial do Corpo (Body Frame)
            "u": 0.0, "v": 0.0, "w": 0.0,  # Linear (Surge, Sway, Heave)
            "p": 0.0, "q": 0.0, "r": 0.0   # Angular (Roll rate, Pitch rate, Yaw rate)
        }

        # -----------------------------------------------------------
        # 2. Espaço de Ação (6 Dimensões: Forças X,Y,Z e Torques K,M,N)
        # -----------------------------------------------------------
        # Definindo limites nominais de força (ex: +/- 50 Newtons)
        # O agente deve aprender a operar dentro desses limites.
        self.action_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(6,),
            dtype=np.float32,
        )

        # -----------------------------------------------------------
        # 3. Espaço de Observação (Alinhado com o Estado)
        # -----------------------------------------------------------
        self.observation_space = spaces.Dict(
            {
                k: spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
                for k in self.state.keys()
            }
        )
        
        self.dt = 0.1  # Time step da simulação
        self.render_mode = render_mode

    def _get_obs(self):
        """Helper para converter o estado interno no formato de observação do Gym"""
        return {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Zera todo o estado
        for key in self.state:
            self.state[key] = 0.0

        # Reseta a dinâmica (ondas JONSWAP e filtros)
        self.dynamics.reset()
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Passo de dinâmica física (atualiza self.state in-place)
        self.dynamics.step(self.state, action)
        
        obs = self._get_obs()
        
        # Cálculo da recompensa (retorna float)
        reward = self.reward_fn.get_reward(obs)

        # -----------------------------------------------------------
        # 4. Condições de Término (Terminated)
        # -----------------------------------------------------------
        terminated = False
        
        # Limite de profundidade (ex: superfície ou fundo muito fundo)
        if abs(self.state["z"]) > 20.0:
            terminated = True
            
        # Limite horizontal (box de operação)
        if abs(self.state["x"]) > 30.0 or abs(self.state["y"]) > 30.0:
            terminated = True
            
        # Limite de Estabilidade (Capotamento)
        # Se o ROV inclinar mais de ~85 graus (1.5 rad), encerra o episódio
        if abs(self.state["roll"]) > 1.5 or abs(self.state["pitch"]) > 1.5:
            terminated = True

        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        self.renderer.render(self.model_path)

    def step_sim(self):
        # Nota: O renderer precisará ser capaz de ler as chaves novas (x,y,z,roll,pitch,yaw)
        # Caso o renderer antigo espere 'theta', isso pode gerar erro visual,
        # mas a lógica da simulação está correta.
        self.renderer.step_sim(self.state)