import os
import pickle as pk
import matplotlib.pyplot as plt

HMR_OUTPUT_PATH = '/home/gilles/Desktop/ORCV_Project/orcv-project/hmr_outputs'

def get_hmr_outputs(tool="scythe", videoname="scythe_0001"):
    """
    Returns the HMR output for a given motion and video reference
    """

    
    tool_path = os.path.join(HMR_OUTPUT_PATH, tool)

    assert videoname in os.listdir(tool_path)
    
    hmr_filepath = os.path.join(tool_path, videoname, "hmr", "hmr.pkl")

    with open(hmr_filepath, 'r') as f:
        data = pk.load(f)

    return data


def print_bothposes(njoints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

	# For each set of style and range settings, plot n random points in the box
	# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

    settings = [('r', 'o'), ('b', 'o')]
    for i in range(2):
        c, m = settings[i]

        xs = njoints[i][:, 0]
        ys = njoints[i][:, 1]
        zs = njoints[i][:, 2]

        ax.scatter(xs, ys, zs, c=c, marker=m)
        
        for ind in range(njoints[i].shape[0]):
            ax.text(xs[ind], ys[ind], zs[ind], str(ind), color=c)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()
    return
    
if __name__ == "__main__":
    from forward_kinematics_smpl import getJointPositionsFromSMPL
    from forward_kinematics_pybullet import getJointPositionsFromHum3d
    import numpy as np

    hmrdata = get_hmr_outputs()
    thetas = hmrdata['thetas']

    theta = thetas[0][3:75]
    beta = thetas[0][75:]
    SMPLPose = getJointPositionsFromSMPL(beta, theta)

    x0 = np.zeros((44))

    x0[4] = 1.
    x0[8] = 1.
    x0[12] = 1.
    x0[16] = 1.
    x0[21] = 1.
    x0[25] = 1.
    x0[30] = 1.
    x0[35] = 1.
    x0[39] = 1.

    Humanoid3Pose = getJointPositionsFromHum3d(x0)


    print_bothposes([SMPLPose])