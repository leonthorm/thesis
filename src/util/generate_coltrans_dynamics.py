import os

import numpy as np
from scipy.spatial.transform import Rotation as R

colors = [
    "0.1 0.8 0.1 1",
    "0.8 0.1 0.1 1",
    "0.1 0.1 0.8 1",
    "0.8 0.8 0.1 1",
    "0.1 0.8 0.8 1",
    "0.8 0.1 0.8 1",
    "0.8 0.8 0.8 1",
    "0.5 0.5 0.5 1",
    "0.1 0.5 0.5 1",
    "0.5 0.1 0.5 1",
    "0.5 0.5 0.1 1",
    "0.2 0.3 0.4 1",
    "0.4 0.3 0.2 1",
    "0.3 0.2 0.4 1",
    "0.4 0.2 0.3 1",
    "0.2 0.4 0.3 1",
    "0.3 0.4 0.2 1",
    "0.9 0.9 0.1 1",
    "0.1 0.9 0.9 1",
]
class MuJoCoSceneGenerator:
    def __init__(self, scene_config):
        self.config = scene_config

    #example config
    # {
    #             "attach_at_site": "payload_s",
    #             "model": "blocks/cf2.xml",
    #             "rope_length": 0.2,
    #             "rope_bodies": 20,
    #             "rope_mass": 0.005,
    #             "rope_damping": 0.00001,
    #             "rope_color_rgba": "0.1 0.1 0.8 1",
    #             "quad_attachment_site": "quad_attachment",
    #             "quad_attachment_offset": [0, 0, 0],
    #         }

    def generate_rope(self, quad_config, quad_id):
        # retrieve rope parameters from the config (with defaults)
        rope_length     = quad_config.get("rope_length", 0.2)
        rope_bodies     = quad_config.get("rope_bodies", 10)
        rope_mass_total = quad_config.get("rope_mass", 0.005)
        rope_damping    = quad_config.get("rope_damping", 0.00001)
        rope_color_rgba = quad_config.get("rope_color_rgba", "0.1 0.8 0.1 1")

        # Compute per-segment values
        delta     = rope_length / rope_bodies          # offset for each nested body
        halfDelta = delta / 2                           # used for geom pos and capsule half-length
        mass_per_body = rope_mass_total / rope_bodies   # mass per geom

        # Start building the XML string
        rope_xml_lines = []
        # first body
        rope_xml_lines.append(f'<body name="q{quad_id}_rope_B_first">')
        rope_xml_lines.append(f'    <geom name="q{quad_id}_rope_G0" size="0.002 {halfDelta}" pos="{halfDelta} 0 0"')
        rope_xml_lines.append(f'        quat="0.707107 0 -0.707107 0" type="capsule" condim="1"')
        rope_xml_lines.append(f'        mass="{mass_per_body}" rgba="{rope_color_rgba}" />')
        rope_xml_lines.append(f'    <site name="q{quad_id}_rope_S_first" pos="0 0 0" group="3" />')
        rope_xml_lines.append(f'    <plugin instance="compositerope{quad_id}_" />')
        
        # intermediate bodies (if any)
        # For bodies 1 to rope_bodies-2, we nest them inside the previous body
        for i in range(1, rope_bodies - 1):
            rope_xml_lines.append(f'    <body name="q{quad_id}_rope_B_{i}" pos="{delta} 0 0">')
            # rope_xml_lines.append(f'         <joint name="q{quad_id}_rope_J_{i}" pos="0 0 0" type="ball" group="3"')
            # rope_xml_lines.append(f'             actuatorfrclimited="false" damping="{rope_damping}" />')
            rope_xml_lines.append(f'         <geom name="q{quad_id}_rope_G{i}" size="0.002 {halfDelta}" pos="{halfDelta} 0 0"')
            rope_xml_lines.append(f'             quat="0.707107 0 -0.707107 0" type="capsule" condim="1"')
            rope_xml_lines.append(f'             mass="{mass_per_body}" rgba="{rope_color_rgba}" />')
            rope_xml_lines.append(f'         <plugin instance="compositerope{quad_id}_" />')
        
        # last body: close with a site and plugin
        rope_xml_lines.append(f'    <body name="q{quad_id}_rope_B_last" pos="{delta} 0 0">')
        # rope_xml_lines.append(f'         <joint name="q{quad_id}_rope_J_last" pos="0 0 0" type="ball" group="3"')
        # rope_xml_lines.append(f'             actuatorfrclimited="false" damping="{rope_damping}" />')
        rope_xml_lines.append(f'         <geom name="q{quad_id}_rope_G_last" size="0.002 {halfDelta}" pos="{halfDelta} 0 0"')
        rope_xml_lines.append(f'             quat="0.707107 0 -0.707107 0" type="capsule" condim="1"')
        rope_xml_lines.append(f'             mass="{mass_per_body}" rgba="{rope_color_rgba}" />')
        rope_xml_lines.append(f'         <site name="q{quad_id}_rope_S_last" pos="{delta} 0 0" group="3" />')
        rope_xml_lines.append(f'         <plugin instance="compositerope{quad_id}_" />')
        
        rope = "\n".join(rope_xml_lines)

        rope_close = ""

        # close all open <body> tags.
        # We opened one tag for the first body, then one for every intermediate and one for the last body.
        # Total open bodies = rope_bodies (first + (rope_bodies-2) intermediates + last)
        for _ in range(rope_bodies):
            rope_close += "</body>\n"
        
        return rope, rope_close

    def rope_lenghth_and_quat(self, quad_config):
        quad_start_position = np.array(quad_config["quad_position"][0])
        payload_start_position  = np.array(self.config["payload_position"][0])

        rope_length = np.linalg.norm(quad_start_position - payload_start_position)

        def compute_rotation(x, y, order='xyz', degrees=True):
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            # Compute direction vector
            direction = y - x
            direction /= np.linalg.norm(direction)  # Normalize

            # Assume reference direction along x-axis
            ref_dir = np.array([1, 0, 0])

            # Compute rotation axis and angle
            axis = np.cross(ref_dir, direction)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-6:  # If vectors are almost aligned
                if np.allclose(ref_dir, direction):
                    return np.array([0, 0, 0])
                else:
                    return np.array([0, 180, 0])  # 180-degree flip

            axis /= axis_norm  # Normalize rotation axis
            angle = np.arccos(np.clip(np.dot(ref_dir, direction)))  # Compute angle

            # Convert axis-angle to rotation matrix
            rot = R.from_rotvec(axis * angle)

            # Convert rotation matrix to Euler angles
            quaternions = rot.as_quat(scalar_first=True)

            return quaternions, rot.inv().as_quat(scalar_first=True)

        rope_quaternions, quad_quaternions = compute_rotation(payload_start_position , quad_start_position)

        return rope_length, rope_quaternions, quad_quaternions

    def generate_quad(self, quad_config):
        id = quad_config["id"]

        quad_config["rope_length"], quad_config["rope_quaternions"], quad_config["quad_quaternions"] = self.rope_lenghth_and_quat(quad_config)
        quad_header = f"""
            <!-- Quad {id} -->
            <body name="q{id}_rope_chain" pos="0 0 0.01" quat="{quad_config["rope_quaternions"][0]} {quad_config["rope_quaternions"][1]} {quad_config["rope_quaternions"][2]} {quad_config["rope_quaternions"][3]} ">
            """
        quad = f"""
            <body
                name="q{id}_cf2"
                childclass="cf2"
                pos="0.0157895 0 0"
                quat="{quad_config["quad_quaternions"][0]} {quad_config["quad_quaternions"][1]} {quad_config["quad_quaternions"][2]} {quad_config["quad_quaternions"][3]} ">
                <inertial
                    pos="0 0 0"
                    mass="0.0356"
                    diaginertia="16.571710e-6 16.655602e-6 16.571710e-6" />
                <joint
                    name="q{id}_joint"
                    type="ball"
                    pos="0 0 0"
                    limited="false"
                    damping="1e-05" />
                <geom
                    class="visual"
                    material="propeller_plastic"
                    mesh="cf2_0" />
                <geom
                    class="visual"
                    material="medium_gloss_plastic"
                    mesh="cf2_1" />
                <geom
                    class="visual"
                    material="polished_gold"
                    mesh="cf2_2" />
                <geom
                    class="visual"
                    material="polished_plastic"
                    mesh="cf2_3" />
                <geom
                    class="visual"
                    material="burnished_chrome"
                    mesh="cf2_4" />
                <geom
                    class="visual"
                    material="body_frame_plastic"
                    mesh="cf2_5" />
                <geom
                    class="visual"
                    material="white"
                    mesh="cf2_6" />
                <geom
                    class="collision"
                    mesh="cf2_collision_0" />
                <geom
                    class="collision"
                    mesh="cf2_collision_1" />
                <geom
                    class="collision"
                    mesh="cf2_collision_2" />
                <geom
                    class="collision"
                    mesh="cf2_collision_3" />
                <geom
                    class="collision"
                    mesh="cf2_collision_4" />
                <geom
                    class="collision"
                    mesh="cf2_collision_5" />
                <geom
                    class="collision"
                    mesh="cf2_collision_6" />
                <geom
                    class="collision"
                    mesh="cf2_collision_7" />
                <geom
                    class="collision"
                    mesh="cf2_collision_8" />
                <geom
                    class="collision"
                    mesh="cf2_collision_9" />
                <geom
                    class="collision"
                    mesh="cf2_collision_10" />
                <geom
                    class="collision"
                    mesh="cf2_collision_11" />
                <geom
                    class="collision"
                    mesh="cf2_collision_12" />
                <geom
                    class="collision"
                    mesh="cf2_collision_13" />
                <geom
                    class="collision"
                    mesh="cf2_collision_14" />
                <geom
                    class="collision"
                    mesh="cf2_collision_15" />
                <geom
                    class="collision"
                    mesh="cf2_collision_16" />
                <geom
                    class="collision"
                    mesh="cf2_collision_17" />
                <geom
                    class="collision"
                    mesh="cf2_collision_18" />
                <geom
                    class="collision"
                    mesh="cf2_collision_19" />
                <geom
                    class="collision"
                    mesh="cf2_collision_20" />
                <geom
                    class="collision"
                    mesh="cf2_collision_21" />
                <geom
                    class="collision"
                    mesh="cf2_collision_22" />
                <geom
                    class="collision"
                    mesh="cf2_collision_23" />
                <geom
                    class="collision"
                    mesh="cf2_collision_24" />
                <geom
                    class="collision"
                    mesh="cf2_collision_25" />
                <geom
                    class="collision"
                    mesh="cf2_collision_26" />
                <geom
                    class="collision"
                    mesh="cf2_collision_27" />
                <geom
                    class="collision"
                    mesh="cf2_collision_28" />
                <geom
                    class="collision"
                    mesh="cf2_collision_29" />
                <geom
                    class="collision"
                    mesh="cf2_collision_30" />
                <geom
                    class="collision"
                    mesh="cf2_collision_31" />
                <site
                    name="q{id}_imu"
                    pos="0 0 0" />
                <site
                    name="q{id}_thrust1"
                    pos="0.03252 -0.03252 0" />
                <site
                    name="q{id}_thrust2"
                    pos="-0.03252 -0.03252 0" />
                <site
                    name="q{id}_thrust3"
                    pos="-0.03252 0.03252 0" />
                <site
                    name="q{id}_thrust4"
                    pos="0.03252 0.03252 0" />

            </body>          
                                                                                       
"""

        rope, rope_close = self.generate_rope(quad_config, id)
        return quad_header + rope + quad + rope_close + "</body>"
        
    def generate_exclude(self, quad_config):
        id = quad_config["id"]
        exclude = f"""
        <exclude body1="q{id}_rope_B_first" body2="q{id}_rope_B_1" />
        """
        body_num = 1
        while body_num < quad_config["rope_bodies"]-3:
            exclude += f"""
            <exclude body1="q{id}_rope_B_{body_num}" body2="q{id}_rope_B_{body_num+1}" />
            """
            body_num += 1
        exclude += f"""
                    <exclude body1="q{id}_rope_B_{body_num}" body2="q{id}_rope_B_{body_num + 1}" />
                    <exclude body1="q{id}_rope_B_{body_num}" body2="q{id}_rope_B_last" />
                    <exclude body1="q{id}_rope_B_last" body2="payload" />
                    """
        return exclude
        
    def generate_xml(self):
        payload_position = self.config["payload_position"]
        paylaod_mass = self.config["payload_mass"]
        rope_instances = ""
        for (i, quad) in enumerate(self.config["quads"]):
            rope_instances += f'<instance name="compositerope{i}_" /> \n'

        header = f"""
    <mujoco model="CF2 scene">
    <compiler angle="radian" meshdir="assets/" eulerseq="xyz"/>

    <option timestep="0.01" density="1.225" viscosity="1.8e-05" integrator="implicit"/>

    <visual>
        <global azimuth="-20" elevation="-20" ellipsoidinertia="true" />
        <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
    </visual>

    <statistic meansize="0.05" extent="0.2" center="0 0 0.1" />

    <default>
        <default class="cf2">
            <site group="5" />
            <default class="visual">
                <geom type="mesh" contype="0" conaffinity="0" group="2" />
            </default>
            <default class="collision">
                <geom type="mesh" group="3" />
            </default>
        </default>
    </default>

    <extension>
        <plugin plugin="mujoco.elasticity.cable">
            {rope_instances}

        </plugin>
    </extension>

    <custom>
        <text name="composite_rope_" data="rope_rope_" />
    </custom>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
            height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="polished_plastic" rgba="0.631 0.659 0.678 1" />
        <material name="polished_gold" rgba="0.969 0.878 0.6 1" />
        <material name="medium_gloss_plastic" rgba="0.109 0.184 0 1" />
        <material name="propeller_plastic" rgba="0.792 0.82 0.933 1" />
        <material name="white" />
        <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1" />
        <material name="burnished_chrome" rgba="0.898 0.898 0.898 1" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
            reflectance="0.2" />
        <mesh name="cf2_0" file="cf2_0.obj" />
        <mesh name="cf2_1" file="cf2_1.obj" />
        <mesh name="cf2_2" file="cf2_2.obj" />
        <mesh name="cf2_3" file="cf2_3.obj" />
        <mesh name="cf2_4" file="cf2_4.obj" />
        <mesh name="cf2_5" file="cf2_5.obj" />
        <mesh name="cf2_6" file="cf2_6.obj" />
        <mesh name="cf2_collision_0" file="cf2_collision_0.obj" />
        <mesh name="cf2_collision_1" file="cf2_collision_1.obj" />
        <mesh name="cf2_collision_2" file="cf2_collision_2.obj" />
        <mesh name="cf2_collision_3" file="cf2_collision_3.obj" />
        <mesh name="cf2_collision_4" file="cf2_collision_4.obj" />
        <mesh name="cf2_collision_5" file="cf2_collision_5.obj" />
        <mesh name="cf2_collision_6" file="cf2_collision_6.obj" />
        <mesh name="cf2_collision_7" file="cf2_collision_7.obj" />
        <mesh name="cf2_collision_8" file="cf2_collision_8.obj" />
        <mesh name="cf2_collision_9" file="cf2_collision_9.obj" />
        <mesh name="cf2_collision_10" file="cf2_collision_10.obj" />
        <mesh name="cf2_collision_11" file="cf2_collision_11.obj" />
        <mesh name="cf2_collision_12" file="cf2_collision_12.obj" />
        <mesh name="cf2_collision_13" file="cf2_collision_13.obj" />
        <mesh name="cf2_collision_14" file="cf2_collision_14.obj" />
        <mesh name="cf2_collision_15" file="cf2_collision_15.obj" />
        <mesh name="cf2_collision_16" file="cf2_collision_16.obj" />
        <mesh name="cf2_collision_17" file="cf2_collision_17.obj" />
        <mesh name="cf2_collision_18" file="cf2_collision_18.obj" />
        <mesh name="cf2_collision_19" file="cf2_collision_19.obj" />
        <mesh name="cf2_collision_20" file="cf2_collision_20.obj" />
        <mesh name="cf2_collision_21" file="cf2_collision_21.obj" />
        <mesh name="cf2_collision_22" file="cf2_collision_22.obj" />
        <mesh name="cf2_collision_23" file="cf2_collision_23.obj" />
        <mesh name="cf2_collision_24" file="cf2_collision_24.obj" />
        <mesh name="cf2_collision_25" file="cf2_collision_25.obj" />
        <mesh name="cf2_collision_26" file="cf2_collision_26.obj" />
        <mesh name="cf2_collision_27" file="cf2_collision_27.obj" />
        <mesh name="cf2_collision_28" file="cf2_collision_28.obj" />
        <mesh name="cf2_collision_29" file="cf2_collision_29.obj" />
        <mesh name="cf2_collision_30" file="cf2_collision_30.obj" />
        <mesh name="cf2_collision_31" file="cf2_collision_31.obj" />
    </asset>

    <worldbody>
        <geom name="floor" pos="0 0 -1.5" size="0 0 0.05" type="plane" material="groundplane" />
        <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
        <body name="payload" pos="{payload_position[0][0]} {payload_position[0][1]} {payload_position[0][2]} ">
            <camera name="track" pos="-1 0 0.5" quat="0.601501 0.371748 -0.371748 -0.601501"
                mode="trackcom" />
            <joint name="payload_joint" type="free" actuatorfrclimited="false" />
            <geom name="payload_geom" size="0.007 0.01" type="cylinder" mass="{paylaod_mass}" rgba="0.8 0.8 0.8 1" />
            <site name="payload_s" pos="0 0 0.01" />
"""

        quads= ""
        for (i, quad) in enumerate(self.config["quads"]):
            q = quad
            q["id"] = i
            pi = 3.14159
            yaw_angle = (2*pi / len(self.config["quads"])) * i - pi
            q["yaw_angle"] = yaw_angle
            quads += self.generate_quad(q)
        quads += "</body>"
        pos_d_sites = ""
        if self.config["generate_path"]:
            for t in range(0, len(self.config["payload_position"]), 5):
                payload_position = self.config["payload_position"][t]
                pos_d_sites += f"""
                <site pos="{payload_position[0]} {payload_position[1]} {payload_position[2]}" euler="0 0 0" rgba="29 131 41 1" />
                """

                for quad_idx, quad in enumerate(self.config["quads"]):
                    quad_position = quad["quad_position"][t]
                    pos_d_sites += f"""
                    <site pos="{quad_position[0]} {quad_position[1]} {quad_position[2]}" euler="0 0 0" rgba="29 131 41 0.7"/>
                    """
        end_worldbody = """
                    
                </worldbody>
                """
        exclude = """ 
            <contact>
            """
        for (i, quad) in enumerate(self.config["quads"]):
            q = quad
            q["id"] = i
            exclude += self.generate_exclude(q)
        exclude += "</contact>"
        actuators = """
        <actuator>
        """
        for quad in self.config["quads"]:
            for i in range(1, 5):
                gear = "0 0 1 0 0 6e-06" if i in [2, 4] else "0 0 1 0 0 -6e-06"
                actuators += f'<general name="q{quad["id"]}_thrust{i}" class="cf2" site="q{quad["id"]}_thrust{i}" ctrlrange="0 0.14" gear="{gear}" />\n'

        actuators += "</actuator>"
        sensor = """
                <sensor>
                    <gyro site="q0_imu" name="body_gyro" />
                    <accelerometer site="q0_imu" name="body_linacc" />
                    <framequat objtype="site" objname="q0_imu" name="body_quat" />
                </sensor>
        """
        end = "</mujoco>"

        return header + quads + pos_d_sites + end_worldbody +  exclude + actuators + sensor + end



def generate_dynamics_xml_from_start(file_name, n_quads, quad_start_pos, cable_lengths, payload_start_pos, generate_path=False):

    scene_config = {
        "payload": "blocks/payload.xml",
        "payload_mass": 0.01,
        "payload_position": payload_start_pos,
        "scene": "blocks/scene.xml",
        "quad_prefix": "q",
        "quads": [],
        "generate_path":generate_path,
    }
    for i in range(n_quads):
        scene_config["quads"].append(
            {
                "attach_at_site": "payload_s",
                "model": "blocks/cf2.xml",
                # "rope_length": 0.5,
                "rope_bodies": 25,
                "rope_mass": 0.0000001,
                "rope_damping": 0.00001,
                "rope_color_rgba": "0.1 0.8 0.1 1",
                "quad_attachment_site": "quad_attachment",
                "quad_attachment_offset": [0, 0, 0],
                "quad_position": quad_start_pos[i],

            }
        )
    generator = MuJoCoSceneGenerator(scene_config)
    dynamics_xml = generator.generate_xml()
    output_file = os.path.join(os.path.dirname(__file__), f"../dynamics/{file_name}")
    with open(output_file, "w") as f:
        f.write(dynamics_xml)
    print(f"Full mujoco xml saved to {output_file}")
    return output_file