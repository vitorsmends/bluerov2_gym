import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

# Nota: O import bluerov2_gym não é estritamente necessário aqui dentro da classe,
# mas mantive caso você use constantes dele.

class BlueRovRenderer:

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        # Inicializa o visualizador apenas se o modo for human
        if self.render_mode == "human":
            self.vis = meshcat.Visualizer()
            try:
                self.vis.open()
            except Exception as e:
                print(f"Aviso: Não foi possível abrir o navegador automaticamente: {e}")
                print(f"Acesse a URL: {self.vis.url()}")

    def render(self, model_path):
        if self.render_mode != "human":
            return
            
        # 1. Superfície da Água
        water_surface = g.Box([30, 30, 0.01])
        water_material = g.MeshPhongMaterial(
            color=0x2389DA, opacity=0.3, transparent=True, side="DoubleSide"
        )
        self.vis["water_surface"].set_object(water_surface, water_material)

        # 2. Volume da Água (Fundo)
        # Corrigido: dimensões positivas [30, 30, 50]
        water_volume = g.Box([30, 30, 50])
        water_volume_material = g.MeshPhongMaterial(
            color=0x1A6B9F, opacity=0.2, transparent=True
        )
        # Translada para baixo para cobrir a área de profundidade
        water_volume_transform = tf.translation_matrix([0, 0, -25])
        self.vis["water_volume"].set_object(water_volume, water_volume_material)
        self.vis["water_volume"].set_transform(water_volume_transform)
        
        print("Carregando modelo 3D de:", model_path)
        
        # 3. O Veículo (BlueROV2)
        try:
            self.vis["vessel"].set_object(
                g.DaeMeshGeometry.from_file(model_path),
                g.MeshLambertMaterial(color=0x0000FF, wireframe=False),
            )
        except Exception as e:
            print(f"Erro ao carregar o arquivo DAE: {e}")
            # Fallback para um cubo se o arquivo não existir
            self.vis["vessel"].set_object(g.Box([0.45, 0.33, 0.25]))

        # 4. O Chão
        ground = g.Box([30, 30, 0.01])
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -20]) # Ajustado para limite Z=20
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)

    def step_sim(self, state):
        self.state = state
        if self.render_mode != "human":
            return

        # 1. Extração da Posição
        x = self.state["x"].item() if isinstance(self.state["x"], np.ndarray) else self.state["x"]
        y = self.state["y"].item() if isinstance(self.state["y"], np.ndarray) else self.state["y"]
        z = self.state["z"].item() if isinstance(self.state["z"], np.ndarray) else self.state["z"]
        
        # 2. Extração da Atitude (Euler Angles)
        # O código antigo usava 'theta', agora usamos roll, pitch, yaw
        phi = self.state["roll"].item() if isinstance(self.state["roll"], np.ndarray) else self.state["roll"]
        theta = self.state["pitch"].item() if isinstance(self.state["pitch"], np.ndarray) else self.state["pitch"]
        psi = self.state["yaw"].item() if isinstance(self.state["yaw"], np.ndarray) else self.state["yaw"]

        # 3. Construção da Matriz de Transformação (4x4)
        transform_matrix = np.eye(4)

        # Matriz de Rotação 3D (Z-Y-X convention / Yaw-Pitch-Roll)
        # R = Rz(psi) * Ry(theta) * Rx(phi)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_phi, s_phi = np.cos(phi), np.sin(phi)

        R = np.array([
            [c_psi * c_th,  c_psi * s_th * s_phi - s_psi * c_phi,  c_psi * s_th * c_phi + s_psi * s_phi],
            [s_psi * c_th,  s_psi * s_th * s_phi + c_psi * c_phi,  s_psi * s_th * c_phi - c_psi * s_phi],
            [-s_th,         c_th * s_phi,                          c_th * c_phi]
        ])

        # Aplica rotação e translação
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = [x, y, z]

        # Atualiza o objeto no Meshcat
        self.vis["vessel"].set_transform(transform_matrix)