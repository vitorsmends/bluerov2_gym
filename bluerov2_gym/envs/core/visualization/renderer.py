import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np


class BlueRovRenderer:
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.vis = None
        self.model_loaded = False

        if self.render_mode == "human":
            self.vis = meshcat.Visualizer()

            try:
                self.vis.open()
            except Exception as e:
                print(f"Aviso: não foi possível abrir o navegador automaticamente: {e}")
                print(f"Acesse a URL: {self.vis.url()}")

    def _scalar(self, value):
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

    def render(self, model_path=None):
        if self.render_mode != "human":
            return

        self._create_world()
        self._load_vehicle(model_path)

    def _create_world(self):
        # Superfície da água em z = 0
        water_surface = g.Box([30.0, 30.0, 0.01])
        water_surface_material = g.MeshPhongMaterial(
            color=0x2389DA,
            opacity=0.30,
            transparent=True,
            side="DoubleSide",
        )

        self.vis["water_surface"].set_object(
            water_surface,
            water_surface_material,
        )
        self.vis["water_surface"].set_transform(
            tf.translation_matrix([0.0, 0.0, 0.0])
        )

        # Volume visual da água abaixo da superfície
        water_volume = g.Box([30.0, 30.0, 40.0])
        water_volume_material = g.MeshPhongMaterial(
            color=0x1A6B9F,
            opacity=0.15,
            transparent=True,
        )

        self.vis["water_volume"].set_object(
            water_volume,
            water_volume_material,
        )
        self.vis["water_volume"].set_transform(
            tf.translation_matrix([0.0, 0.0, -20.0])
        )

        # Fundo
        ground = g.Box([30.0, 30.0, 0.02])
        ground_material = g.MeshPhongMaterial(
            color=0x808080,
            side="DoubleSide",
        )

        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(
            tf.translation_matrix([0.0, 0.0, -20.0])
        )

        # Eixos de referência
        self.vis["world_axes"].set_object(g.triad(1.0))

    def _load_vehicle(self, model_path=None):
        if model_path is not None:
            print("Carregando modelo 3D de:", model_path)

        try:
            if model_path is None:
                raise FileNotFoundError("model_path não informado.")

            self.vis["vessel"].set_object(
                g.DaeMeshGeometry.from_file(model_path)
            )

        except Exception as e:
            print(f"Aviso: não foi possível carregar o arquivo DAE: {e}")
            print("Usando geometria simplificada do BlueROV2.")

            self.vis["vessel"].set_object(
                g.Box([0.40, 0.25, 0.10]),
                g.MeshLambertMaterial(
                    color=0x0000FF,
                    wireframe=False,
                ),
            )

        self.model_loaded = True

    def step_sim(self, state):
        if self.render_mode != "human":
            return

        if not self.model_loaded:
            self.render(model_path=None)

        x = self._scalar(state["x"])
        y = self._scalar(state["y"])
        z = self._scalar(state["z"])

        roll = self._scalar(state["roll"])
        pitch = self._scalar(state["pitch"])
        yaw = self._scalar(state["yaw"])

        T = self._pose_to_transform(x, y, z, roll, pitch, yaw)

        self.vis["vessel"].set_transform(T)

    def _pose_to_transform(self, x, y, z, roll, pitch, yaw):
        """
        Builds homogeneous transform from position and ZYX Euler angles.

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        """

        c_psi, s_psi = np.cos(yaw), np.sin(yaw)
        c_th, s_th = np.cos(pitch), np.sin(pitch)
        c_phi, s_phi = np.cos(roll), np.sin(roll)

        R = np.array([
            [
                c_psi * c_th,
                c_psi * s_th * s_phi - s_psi * c_phi,
                c_psi * s_th * c_phi + s_psi * s_phi,
            ],
            [
                s_psi * c_th,
                s_psi * s_th * s_phi + c_psi * c_phi,
                s_psi * s_th * c_phi - c_psi * s_phi,
            ],
            [
                -s_th,
                c_th * s_phi,
                c_th * c_phi,
            ],
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T