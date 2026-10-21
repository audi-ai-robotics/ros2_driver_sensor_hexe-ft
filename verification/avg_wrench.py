"""Average the UR-side wrench (what force_mode sees) and the raw HEX-E over a few seconds.

Usage: python3 avg_wrench.py [seconds=3] [ns=Peach]
The UR topic only carries the HEX-E while the External Control program runs (ft_rtde_input_enable
lives in that program); otherwise it is the UR built-in sensor. The last line says which.
"""
import sys, time, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, DurabilityPolicy
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
ns = sys.argv[2] if len(sys.argv) > 2 else 'Peach'
rclpy.init(); n = Node('avg_wrench'); d = {'ur': [], 'hex': []}
arr = lambda m: [m.wrench.force.x, m.wrench.force.y, m.wrench.force.z, m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z]
n.create_subscription(WrenchStamped, f'/{ns}/force_torque_sensor_broadcaster/wrench', lambda m: d['ur'].append(arr(m)), qos_profile_sensor_data)
n.create_subscription(WrenchStamped, f'/{ns}/hex_e_ft_broadcaster/wrench', lambda m: d['hex'].append(arr(m)), qos_profile_sensor_data)
prog = []
n.create_subscription(Bool, f'/{ns}/io_and_status_controller/robot_program_running', lambda m: prog.append(m.data),
                      QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
t = time.time()
while time.time() - t < dur: rclpy.spin_once(n, timeout_sec=0.05)
np.set_printoptions(precision=3, suppress=True)
for k, name in (('ur', 'UR topic (force_mode input)'), ('hex', 'raw HEX-E')):
    a = np.array(d[k])
    if len(a): print(f'{name:28s} n={len(a):5d}  mean F={a[:, :3].mean(0)} T={a[:, 3:].mean(0)}  std F={a[:, :3].std(0)}')
    else: print(f'{name:28s} no data')
print('robot program running:', prog[-1] if prog else 'unknown',
      '-> UR topic is the HEX-E' if prog and prog[-1] else '-> UR topic is the BUILT-IN sensor (start the program on the pendant)')
