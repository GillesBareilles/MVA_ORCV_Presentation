

## HMR to DeepMimic

Conversion from HMR output (SMPL format) to DeepMimic input (Humanoid3 format). Conversion pipeline is as follows: SMPL pose (rotation) -> SMPL pose (3d positions) -> Humanoid3 pose (3d positions) -> Humanoid3 pose (rotation). First step is done by forward kinematics, as suggested and implemented by Zongmian Li (inria Paris). Next steps are done by one-to-one joint correspondance, then rotation computation.

Motion files were produced by running the `hmr_to_humanoid3.py` script, which relies on the `forward_kinematics_pybullet.py` and `inverse_kinematics_deepmimic.py`. 

The remainder of the files are either utils functions or implementation of an other approach using forward kinematics for both SMPL and Humanoid3 and optimisation. It yielded unsatisfactory results, both in quality of mapping and execution time.