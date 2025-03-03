import os
import sys
import yaml
import numpy as np

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


class QuadSceneGenerator:
    def __init__(self, scene_config):
        # Apply defaults to the scene config
        defaults = {
            "payload_connection": "tendon",  # ["cable", "tendon", "none"]
            "options": {
                "timestep": 0.004,
                "density": 1.2,  # air density
                "viscosity": 0.00002,  # air viscosity
                "integrator": "Euler",
                "gravity": "0 0 -9.81",
                "wind": "0 0 0",
            },
            "compiler": {
                "angle": "radian",
                "meshdir": "assets/",
                "discardvisual": "false"
            },
            "goal": {
                "pos": [0, 0, 0.5],
                "size": 0.02,
                "rgba": "1 0 0 0.8"
            },
            "payload": {
                "mass": 0.01,
                "geom_type": "cylinder",
                "size": [0.007, 0.01],
                "start_pos": False,  # [0, 0, 0.1] or False
                "start_euler": [0, 0, 0],
                "rgba": "0.8 0.8 0.8 1",
                "attach_sites": [
                    {
                        "name": "attach_site_1",
                        "pos": [0, 0, 0.01]
                    },
                    {
                        "name": "attach_site_2",
                        "pos": [0, 0, -0.01]
                    }
                ]
            },
            "quads": []
        }
        # Merge the provided scene_config into defaults (shallow merge)
        merged = defaults.copy()
        merged.update(scene_config)
        self.config = merged

    # example config
    # {
    #             "attach_at_site": "payload_s",
    #             "model": "blocks/cf2.xml",
    #             "cable_length": 0.2,
    #             "cable_bodies": 20,
    #             "cable_mass": 0.005,
    #             "cable_damping": 0.00001,
    #             "cable_rgba": "0.1 0.1 0.8 1",
    #             "quad_attachment_site": "quad_attachment",
    #             "quad_attachment_offset": [0, 0, 0],
    #         }

    def generate_cable(self, quad_config, quad_id):
        # retrieve rope parameters from the config (with defaults)
        cable_config = quad_config["cable"]
        cable_length = cable_config.get("length", 0.2)
        cable_bodies = cable_config.get("bodies", 20)
        cable_mass_total = cable_config.get("mass", 0.001)
        cable_damping = cable_config.get("damping", 0.00001)
        cable_thickness = cable_config.get("thickness", 0.002)

        cable_rgba = cable_config.get("rgba", colors[quad_id % len(colors)])

        # Compute per-segment values
        delta = cable_length / cable_bodies  # offset for each nested body
        halfDelta = delta / 2  # used for geom pos and capsule half-length
        mass_per_body = cable_mass_total / cable_bodies  # mass per geom

        # Build the rope XML using multi-line strings
        first_body = f"""
        <body name="q{quad_id}_cable_B_first">
            <joint name="q{quad_id}_cable_J_first" pos="0 0 0" type="ball" group="3"
                actuatorfrclimited="false" damping="{cable_damping}" />
            <geom name="q{quad_id}_cable_G0" size="{cable_thickness} {halfDelta}" pos="{halfDelta} 0 0"
            quat="0.707107 0 -0.707107 0" type="capsule" condim="1"
            mass="{mass_per_body}" rgba="{cable_rgba}" />
            <site name="q{quad_id}_cable_S_first" pos="0 0 0" group="3" />
            <plugin instance="compositecable{quad_id}_" />
        """
        intermediate_bodies = ""
        for i in range(1, cable_bodies - 1):
            intermediate_bodies += f"""
            <body name="q{quad_id}_cable_B_{i}" pos="{delta} 0 0">
             <joint name="q{quad_id}_cable_J_{i}" pos="0 0 0" type="ball" group="3"
                 actuatorfrclimited="false" damping="{cable_damping}" />
             <geom name="q{quad_id}_cable_G{i}" size="{cable_thickness} {halfDelta}" pos="{halfDelta} 0 0"
                 quat="0.707107 0 -0.707107 0" type="capsule" condim="1"
                 mass="{mass_per_body}" rgba="{cable_rgba}" />
             <plugin instance="compositecable{quad_id}_" />
            """
        last_body = f"""
            <body name="q{quad_id}_cable_B_last" pos="{delta} 0 0">
             <joint name="q{quad_id}_cable_J_last" pos="0 0 0" type="ball" group="3"
                 actuatorfrclimited="false" damping="{cable_damping}" />
             <geom name="q{quad_id}_cable_G_last" size="{cable_thickness} {halfDelta}" pos="{halfDelta} 0 0"
                 quat="0.707107 0 -0.707107 0" type="capsule" condim="1"
                 mass="{mass_per_body}" rgba="{cable_rgba}" />
             <site name="q{quad_id}_cable_S_last" pos="{delta} 0 0" group="3" />
             <plugin instance="compositecable{quad_id}_" />
        """
        rope = first_body + intermediate_bodies + last_body

        cable_close = ""

        # close all open <body> tags.
        # We opened one tag for the first body, then one for every intermediate and one for the last body.
        # Total open bodies = cable_bodies (first + (cable_bodies-2) intermediates + last)
        for _ in range(cable_bodies):
            cable_close += "</body>\n"

        return rope, cable_close

    def val_string(self, dict):
        # turn into = string. Make sure arrays are space-separated
        for key, val in dict.items():
            if isinstance(val, list):
                dict[key] = " ".join(map(str, val))
        return " ".join([f'{key}="{val}"' for key, val in dict.items()])

    def generate_quad(self, quad_config):
        id = quad_config["id"]
        yaw_angle = quad_config.get("yaw_angle", 0)
        quad_header = f"""
            <!-- Quad {id} -->
            <body name="q{id}_cable_chain" pos="0 0 0.01" euler="0 0 {yaw_angle}">
            """
        attachment_offset = "0 0 0"
        if quad_config["payload_connection"] != "none":
            attachment_offset = f"{quad_config['cable']['attachment_offset'][0]} {quad_config['cable']['attachment_offset'][1]} {quad_config['cable']['attachment_offset'][2] - 0.0015}"
        pos = "0.0157895 0 0"
        if quad_config["payload_connection"] == "cable":
            quad_payload_joint = f"""
             <joint
                    type="ball"
                    pos="0 0 0"
                    limited="false"
                    damping="0.00001" 
                    name="q{id}_joint"/>
            """
        else:
            pos = f"{quad_config['start_pos'][0]} {quad_config['start_pos'][1]} {quad_config['start_pos'][2]}"
            quad_payload_joint = f"""
            <joint type="free" actuatorfrclimited="false" name="q{id}_joint"/>
            """

        quad = f"""
        <body
            name="q{id}_container"
            pos="{pos}"
            >        
            {quad_payload_joint}
            <site name="q{id}_attachment" pos="0 0 0" group="5"/>
            <body
                name="q{id}_cf2"
                childclass="cf2"
                pos="{attachment_offset}"
                >
                <inertial
                    pos="0 0 0"
                    mass="0.034"
                    diaginertia="1.65717e-05 1.66556e-05 2.92617e-05" />

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
                    pos="0.032527 -0.032527 0" />
                <site
                    name="q{id}_thrust2"
                    pos="-0.032527 -0.032527 0" />
                <site
                    name="q{id}_thrust3"
                    pos="-0.032527 0.032527 0" />
                <site
                    name="q{id}_thrust4"
                    pos="0.032527 0.032527 0" />

            </body>    
        </body>      

"""
        if quad_config["payload_connection"] == "cable":
            cable, cable_close = self.generate_cable(quad_config, id)
            return quad_header + cable + quad + cable_close + "</body>"
        else:
            return quad

    def generate_xml(self):

        cable_instances = ""
        for quad in self.config["quads"]:
            cable_instances += f'<instance name="compositecable{quad["id"]}_" /> \n'

        # Dynamically generate quad start sites
        quad_start_sites = ""
        for quad in self.config["quads"]:
            quad_start_sites += f'<site name="q{quad["id"]}_start" pos="{" ".join(map(str, quad["start_pos"]))}" euler="{" ".join(map(str, quad["start_euler"]))}" />\n'

        # Dynamically generate payload start site
        if self.config["payload"]["start_pos"] is False:
            payload_start_site = ""
        else:
            payload_start_site = f'<site name="payload_start" pos="{" ".join(map(str, self.config["payload"]["start_pos"]))}" euler="{" ".join(map(str, self.config["payload"]["start_euler"]))}" />\n'

        payload_connection = self.config.get("payload_connection", "cable")
        if payload_connection == "none":
            payload = ""
        else:
            init_pos = self.config["payload"]["start_pos"]
            type = self.config["payload"]["geom_type"]
            p_size = " ".join(map(str, self.config["payload"]["size"]))
            if init_pos is False:
                # If no start pos is given, average out the quad start positions
                init_pos = np.mean([np.array(quad["start_pos"]) for quad in self.config["quads"]], axis=0) - np.array(
                    [0, 0, 0.1])

            payload = f"""
            <body name="payload" pos="{init_pos[0]} {init_pos[1]} {init_pos[2]}" >
            <camera name="track" pos="-1 0 0.5" quat="0.601501 0.371748 -0.371748 -0.601501"
                mode="trackcom" />
            <joint type="free" actuatorfrclimited="false" damping="0.00001" name="payload_joint"/>
            <geom size="{p_size}" type="{type}" mass="0.001" rgba="0.8 0.8 0.8 1" />
            <site name="payload_s" pos="0 0 0.01" />
            {"</body>" if payload_connection != "cable" else ""}
            """

        header = f"""
    <mujoco>
    <compiler {self.val_string(self.config["compiler"])} />
    <option {self.val_string(self.config["options"])} />

    <visual>
        <global azimuth="-20" elevation="-20" ellipsoidinertia="true" />
        <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
    </visual>

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
            {cable_instances}
        </plugin>
    </extension>

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
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="10 10"
            reflectance="0" />
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
        <geom name="goal_marker" contype="0" conaffinity="0" {self.val_string(self.config["goal"])} />
        <geom name="floor" pos="0 0 -1.5" size="0 0 0.05" type="plane" material="groundplane" />
        <light pos="0 0 1.5" dir="0 0 -1" directional="true" />

        {quad_start_sites}
        <!-- Payload start site-->
        {payload_start_site}

        {payload}
"""
        quads = ""
        for (i, quad) in enumerate(self.config["quads"]):
            # set yaw angle for quad placement to point from init pos to quad start pos
            if payload_connection != "none":
                vec = np.array(quad["start_pos"]) - init_pos
                quad["yaw_angle"] = np.arctan2(vec[1], vec[0])
            quad["payload_connection"] = payload_connection

            quads += self.generate_quad(quad)

        # Dynamically generate tendons for all quads
        tendons = ""
        if payload_connection == "tendon":
            for quad in self.config["quads"]:
                tendon_rgba = quad["cable"].get("rgba", colors[quad["id"] % len(colors)])
                tendons += f"""
                <spatial name="q{quad["id"]}_tendon" limited="true" range="0 {quad["cable"]["length"]}" width="{quad["cable"]["thickness"]}" rgba="{tendon_rgba}">
                    <site site="q{quad["id"]}_attachment" />
                    <site site="payload_s" />
                </spatial>
                """

        # Dynamically generate equality welds for all quads
        equality_welds = ""
        for quad in self.config["quads"]:
            equality_welds += f'<weld site1="q{quad["id"]}_start" site2="q{quad["id"]}_imu" solref="0.01 4" />\n'

        if payload_start_site != "" and payload_connection != "none":
            equality_welds += f'<weld site1="payload_start" site2="payload_s" solref="0.02 5" />\n'

        # Dynamically generate actuator definitions for each quad
        actuators = ""
        for quad in self.config["quads"]:
            for i in range(1, 5):
                gear = "0 0 1 0 0 6e-06" if i in [1, 3] else "0 0 1 0 0 -6e-06"
                actuators += f'<general name="q{quad["id"]}_thrust{i}" class="cf2" site="q{quad["id"]}_thrust{i}" ctrlrange="0 0.14" gear="{gear}" />\n'

        # Dynamically generate sensor definitions for each quad
        sensors = ""
        for quad in self.config["quads"]:
            sensors += f'<gyro site="q{quad["id"]}_imu" name="q{quad["id"]}_gyro" />\n'
            sensors += f'<accelerometer site="q{quad["id"]}_imu" name="q{quad["id"]}_linacc" />\n'
            sensors += f'<framequat objtype="site" objname="q{quad["id"]}_imu" name="q{quad["id"]}_framequat" />\n'
            sensors += f'<force name="q{quad["id"]}_cable_force" site="q{quad["id"]}_attachment" />\n'

        # Dynamically generate contact exclusions for cable bodies of each quad
        contact_exclusions = ""
        if payload_connection == "cable":
            for quad in self.config["quads"]:
                bodies = quad["cable"]["bodies"]
                if bodies > 1:
                    contact_exclusions += f'<exclude body1="q{quad["id"]}_cable_B_first" body2="q{quad["id"]}_cable_B_1" />\n'
                for i in range(1, bodies - 2):
                    contact_exclusions += f'<exclude body1="q{quad["id"]}_cable_B_{i}" body2="q{quad["id"]}_cable_B_{i + 1}" />\n'
                contact_exclusions += f'<exclude body1="q{quad["id"]}_cable_B_{bodies - 2}" body2="q{quad["id"]}_cable_B_last" />\n'
                contact_exclusions += f'<exclude body1="q{quad["id"]}_cable_B_first" body2="payload" />\n'

        end = f"""
        {"</body>" if payload_connection == "cable" else ""}
    </worldbody>

    <contact>
        {contact_exclusions}
    </contact>

    <tendon>
        {tendons}
    </tendon>

    <equality>
        {equality_welds}       
    </equality>

    <actuator>
        {actuators}
    </actuator>

    <sensor>
        {sensors}
    </sensor>
</mujoco>"""
        return header + quads + end

    def generate_xml_from_yaml(self, yaml_filepath):
        with open(yaml_filepath, "r") as f:
            scene_config = yaml.safe_load(f)
        return QuadSceneGenerator(scene_config).generate_xml()

    def generate_random(self):
        # Generate a random scene configuration
        self.config["quads"] = []
        for i in range(4):
            self.config["quads"].append({
                "id": i,
                "cable": {
                    "length": np.random.uniform(0.1, 0.3),
                    "bodies": np.random.randint(10, 30),
                    "mass": 0.001,
                    "damping": 0.00001,
                    "thickness": 0.002,
                    "attachment_offset": [0, 0, 0],
                    "rgba": colors[i % len(colors)]
                },
                "start_pos": [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0.1],
                "start_euler": [0, 0, 0],
                "payload_connection": "cable"
            })
        return self.generate_xml()

def generate_xml_from_start(file_name, n_quads, quad_pos, cable_lengths, payload_pos=False, generate_path=False):
    scene_config = {
        "payload_connection": "cable",  # options: "cable", "tendon", "none"
        "options": {
            "timestep": 0.004,
            "density": 1.2,
            "viscosity": 0.00002,
            "integrator": "Euler",
            "gravity": "0 0 -9.81",
            "wind": "0 0 0",
        },
        "compiler": {
            "angle": "radian",
            "meshdir": "assets/",
            # "discardvisual": False,
        },
        "goal": {
            "pos": [0, 0, 1],
            "size": 0.01,
            "rgba": "1 0 0 0.5",
        },
        "payload": {
            "mass": 0.01,
            "geom_type": "sphere",
            "size": [0.01],
            "start_pos": payload_pos[0],  # array or false; payload start site
            "start_euler": [0, 0, 1],
            "rgba": "0.8 0.8 0.8 1",
            "attach_sites": [
                {
                    "name": "attach_site_1",
                    "pos": [0, 0, 0.01],
                }
            ],
        },
        "quads": [],
    }

    for i in range(n_quads):
        scene_config["quads"].append(
            {
                "id": i,
                "start_pos": quad_pos[i][0],
                "start_euler": [0, 0, 0],
                "cable": {
                    "length": cable_lengths[i],
                    "thickness": 0.003,
                    "bodies": 20,
                    "mass": 0.00001,
                    "quad_site": "q0_attachment",
                    "attachment_offset": [0, 0, 0],
                    "payload_site": "attach_site_1",
                },
            }
        )
    generator = QuadSceneGenerator(scene_config)
    dynamics_xml = generator.generate_xml()
    output_file = os.path.join(os.path.dirname(__file__), f"../dynamics/{file_name}")
    with open(output_file, "w") as f:
        f.write(dynamics_xml)
    print(f"Full mujoco xml saved to {output_file}")
    return output_file
if __name__ == "__main__":
    # If a YAML file path is provided, load config from it.
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
        generator = QuadSceneGenerator({})  # dummy init
        full_xml = generator.generate_xml_from_yaml(yaml_path)
        output_file = yaml_path.replace(".yaml", ".xml")
        with open(output_file, "w") as f:
            f.write(full_xml)
            print(f"Full mujoco xml saved to {output_file}")
            print(f"To visualize the scene, run: \npython -m mujoco.viewer --mjcf={output_file}")
    else:
        print("Usage: python generate_swarm.py <path_to_yaml_config>")

