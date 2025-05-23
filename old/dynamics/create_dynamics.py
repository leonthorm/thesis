# Save XML dynamically
with open("point_mass_py.xml", "w") as f:
        f.write('<mujoco model="point_mass">\n')
        f.write('  <option timestep="0.01" />\n\n')

        # Asset
        f.write('  <asset>\n')
        f.write('    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"\n')
        f.write('     rgb2=".2 .3 .4" width="300" height="300"/>\n')
        f.write('    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0"/>\n')
        f.write('  </asset>\n\n')

        # Worldbody
        f.write('  <worldbody>\n')
        # Uncomment for plane
        # f.write('    <geom size="20 20 .01" type="plane" material="grid"/>\n')
        f.write('    <body name="point_mass" pos="0 0 0">\n')
        f.write('      <joint name="joint_1" type="slide" axis="1 0 0"/>\n')
        f.write('      <joint name="joint_2" type="slide" axis="0 1 0"/>\n')
        f.write('      <joint name="joint_3" type="slide" axis="0 0 1"/>\n')
        f.write('      <geom type="sphere" size="0.1" mass="0.1" rgba="255.0 0.0 0.0 1.0"/>\n')
        f.write('    </body>\n')
        f.write('    <site name="target_state" pos="0.5 0.25 0.5" size="0.02" type="sphere" rgba="0 1 0 1"/>\n')
        for i in range(10):
            f.write(f'    <site name="traj_site_{i}" pos="0 0 0" size="0.02" rgba="255 240 0 1"/>\n')
        f.write('  </worldbody>\n\n')

        # Actuator
        f.write('  <actuator>\n')
        f.write('    <motor joint="joint_1" ctrlrange="-10 10" ctrllimited="true"/>\n')
        f.write('    <motor joint="joint_2" ctrlrange="-10 10" ctrllimited="true"/>\n')
        f.write('    <motor joint="joint_3" ctrlrange="-10 10" ctrllimited="true"/>\n')
        f.write('  </actuator>\n')
        f.write('</mujoco>\n')
