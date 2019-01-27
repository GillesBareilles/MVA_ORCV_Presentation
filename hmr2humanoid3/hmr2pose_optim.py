# -*- coding: utf-8 -*-
# import numpy as np
# #import cPickle as pk
# import pickle as pk
# import pandas as pd
# import math
# import json

import os


from utils import get_hmr_outputs, print_bothposes
from utils_pybullet import display_joints_matplotlib, displayBulletFrames
from forward_kinematics_smpl import getJointPositionsFromSMPL, SMPLPose_to_Hmu3dSpace
from forward_kinematics_pybullet import getJointPositionsFromHum3d
from numdifftools import Jacobian, Hessian
from neighbor import neighboor

import numpy as np
import matplotlib.pyplot as plt
from math import isnan
import json


from optim_simannealing import sim_annealing
from optim_scipy import optimize_scipy
from optim_coorddescent import optim_coorddescent, optim_randcoorddescent, optim_treecoorddescent

param = 0

def distance(x_bullet, smpl_joints):
    pybullet = getJointPositionsFromHum3d(x_bullet, target_joints=smpl_joints)

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



def buildWrite_Hum3dJSON_from_HMRoutput(hmrdata, outfilename):
    thetas = hmrdata['thetas']

    numFrames = thetas.shape[0]
    hum3dData = np.zeros((numFrames, 44))

    ## Treat first pose in detail  
    # Build SMPL 3d pose
    theta, beta = thetas[0][3:75], thetas[0][75:]
    
    joint_3d_position = getJointPositionsFromSMPL(beta, theta)
    joint_3d_position_corr = SMPLPose_to_Hmu3dSpace(joint_3d_position)

    # Fit Humanoid3 vector
    x0 = np.zeros((44))
    for ind_start in [4, 8, 12, 16, 21, 25, 30, 35, 39]:
            x0[ind_start] = 1.
    hum3dData[0, :] = optim_treecoorddescent(x0, joint_3d_position_corr)


    # Then the reset by recurrence
    for ind_frame in range(1, numFrames):
        print ind_frame
        
        # Build SMPL 3d pose
        theta = thetas[ind_frame][3:75]
        beta = thetas[ind_frame][75:]
        
        joint_3d_position = getJointPositionsFromSMPL(beta, theta)
        joint_3d_position_corr = SMPLPose_to_Hmu3dSpace(joint_3d_position)

        # Fit last 
        hum3dData[ind_frame, :] = optim_randcoorddescent(hum3dData[ind_frame-1, :], joint_3d_position_corr)


    ## Write JSON in DeepMimic format
    deepmimicdict = {
        "Loop": "none",
        "Frames": hum3dData.tolist()
    }

    with open("output.json", "w") as f:
        json.dump(deepmimicdict, f)


if __name__ == "__main__":

    hmrdata = get_hmr_outputs()

    buildWrite_Hum3dJSON_from_HMRoutput(hmrdata, "output.json")

    thetas = hmrdata['thetas']


    # Theta is the 85D vector holding [camera, pose, shape] where:
    # -3        : camera is 3D [s, tx, ty]
    # -72 = 3*24: pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # -10       : shape is 10D shape coefficients of SMPL
    cam = thetas[0][0:3]
    theta = thetas[0][3:75]
    beta = thetas[0][75:]


    joint_3d_position = getJointPositionsFromSMPL(beta, theta)
    joint_3d_position_corr = SMPLPose_to_Hmu3dSpace(joint_3d_position)

    x0 = np.zeros((44))
    for ind_start in [4, 8, 12, 16, 21, 25, 30, 35, 39]:
            x0[ind_start] = 1.

    if False:
        fig_kind = "scythe"
        x0 = np.array([ 0., 0.01224597, 0.81401073, 0.01096643, 0.65879193, 0.05304061, -0.74939111, -0.0399104, 0.96979005, 0.18838892, -0.13597595, -0.07434665,0.99822753, 0.01487787, 0.00908718, -0.05690237, 0.96790066, -0.21062716,-0.10093313, -0.09282787, -0.03664607, 0.77579692, 0.05718172, -0.00129648,0.62838501, 0.93725933, -0.32222106, 0.02839909, 0.13004624, 0.32039612,0.9995328, -0.02482968, 0.01192958, -0.01324183, -0.05720162, 0.9280872,0.10006004, -0.00599808, 0.35861702, 0.99242762, 0.09809208, 0.04151842,0.06116843, 0.10694508])
    x0[1:8] = [0.01224597, 0.81401073, 0.01096643, 0.65879193, 0.05304061, -0.74939111, -0.0399104]

    ## Optimize
    # res = optimize_scipy(x0, joint_3d_position)
    # res = sim_annealing(x0, joint_3d_position)
    # res = optim_coorddescent(x0, joint_3d_position_corr)
    # res = optim_randcoorddescent(x0, joint_3d_position_corr)
    res = optim_treecoorddescent(x0, joint_3d_position_corr)