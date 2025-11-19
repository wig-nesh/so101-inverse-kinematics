# SO101 MuJoCo 

robotic arm that *dreams* of reaching things in simulation before attempting it in the real world.

## What is this?

This repo teaches the arm to move using MuJoCo physics engine. 
Think of it as robot yoga—lots of stretching, some keyboard prodding, and the occasional circular motion.

## Start

```bash
cd src
uv pip install -e .
```

Then make the arm do things. 

## Files Worth Knowing

- `robot.py` - The arm's brain (or what passes for one)
- `keyboard_control_rerun.py` - Your new remote control with a beautiful interface ( will only work with X11 )
- `keyboard_control_rerun_wayland.py` - For Wayland 
- `circular_path.py` - Teaching the arm to do circles
- `so101.urdf` - What the arm looks like to MuJoCo

No guarantees it won't get stuck in a corner.
