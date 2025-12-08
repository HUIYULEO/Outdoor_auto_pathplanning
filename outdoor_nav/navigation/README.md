# Navigation Module

Local navigation and control for the outdoor robot, including both offline
planning demos and a script used on the real Hopper robot system.

- `src/trajectory.py` – Pure Pursuit controller for trajectory tracking
- `src/run_full_nav_offline.py` – example pipeline: image → mask → path → direction
- `src/robot_interface.py` – example command script / interface used on the
  actual Hopper robot (Mobotware-based system)
