import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


from math import isnan
from neighbor import neighboor
from forward_kinematics_pybullet import getJointPositionsFromHum3d


def distance(x_bullet, smpl_joints):
    # global param
    # param += 1
    # print param
    # print "x_bullet is"
    # print x_bullet

    # smpl_joints = getJointPositionsFromSMPL(beta, theta)
    pybullet = getJointPositionsFromHum3d(x_bullet, target_joints=smpl_joints)

    # print "SMPL joints positions"
    # print smpl_joints

    # print "\nBullet joints position"
    # print "joints position is:"
    # print pybullet

    # print_bothposes([smpl_joints, pybullet])
    weights = np.ones((15))
    weights[0] = 10
    weights[1] = 3
    weights[2] = 3
    weights /= weights.sum()

    dist = 0.
    dist += weights[0] *  np.linalg.norm( smpl_joints[3 , :] - pybullet[0, :] )**2
    dist += weights[1] *  np.linalg.norm( smpl_joints[9, :] - pybullet[1, :] )**2
    dist += weights[2] *  np.linalg.norm( smpl_joints[15, :] - pybullet[2, :] )**2
    
    dist += weights[3] *  np.linalg.norm( smpl_joints[16, :] - pybullet[3, :] )**2 #left arm
    dist += weights[4] *  np.linalg.norm( smpl_joints[18, :] - pybullet[4, :] )**2
    dist += weights[5] *  np.linalg.norm( smpl_joints[20, :] - pybullet[5, :] )**2
    
    dist += weights[6] *  np.linalg.norm( smpl_joints[17, :] - pybullet[6, :] )**2 # right arm
    dist += weights[7] *  np.linalg.norm( smpl_joints[19, :] - pybullet[7, :] )**2
    dist += weights[8] *  np.linalg.norm( smpl_joints[21, :] - pybullet[8, :] )**2
    
    dist += weights[9] *  np.linalg.norm( smpl_joints[1, :] - pybullet[9, :] )**2 # left leg
    dist += weights[10] *  np.linalg.norm( smpl_joints[4, :] - pybullet[10, :] )**2
    dist += weights[11] *  np.linalg.norm( smpl_joints[7, :] - pybullet[11, :] )**2

    dist += weights[12] *  np.linalg.norm( smpl_joints[2, :] - pybullet[12, :] )**2 # right leg
    dist += weights[13] *  np.linalg.norm( smpl_joints[5, :] - pybullet[13, :] )**2
    dist += weights[14] *  np.linalg.norm( smpl_joints[8, :] - pybullet[14, :] )**2

    print "dist is ", dist
    return dist


def optimize_scipy(x0, joint_3d_position):

    # Unit quaternion constraint
    constraints = []
    for ind_start in [4, 8, 12, 16, 21, 25, 30, 35, 39]:
        f = lambda x : np.linalg.norm(x[ind_start:ind_start+4])**2
        constraints.append({'type':'eq', 'fun': f})

    # def fun_der(x, beta, theta):
    #     return Jacobian(lambda x: distance(x, beta, theta))(x).ravel()

    # def fun_hess(x):
    #     return Hessian(lambda x: distance(x))(x)

    options = {}
    options["disp"] = True
    options["verbose"] = 3
    res = opt.minimize(distance, x0, args = (joint_3d_position), method='Nelder-Mead', bounds=(-2, 2), constraints = constraints, options = options)

    return res
