"""AllenAI ToyBlocks (Isaac Sim / Isaac Lab, single Franka FR3) robot backend.

Select it with ``robot_receiver.py --sim toyblocks``. Only ``backend`` and
``scene``/``engine``/``skills``/``entity`` import Isaac Lab, and only when a
backend is actually built; ``frames`` and ``staging`` are pure Python so they can
be tested anywhere.

Pinned ToyBlocks commit: 931fd7aab1458fd58f4c23d810879ea4826b07d8. The engine
adapter relies on vecgen internals, so update the pin deliberately.
"""
TOYBLOCKS_COMMIT = "931fd7aab1458fd58f4c23d810879ea4826b07d8"
