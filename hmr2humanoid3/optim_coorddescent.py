# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


from math import isnan
from neighbor import neighboor, perturb_angle, perturb_position, perturb_quaternion
from forward_kinematics_pybullet import getJointPositionsFromHum3d



def distance(x_bullet, smpl_joints, jointsMetric = None):
    pybullet = getJointPositionsFromHum3d(x_bullet, target_joints=smpl_joints)

    # weights = np.ones((15))
    # weights[0] = 10
    # weights[1] = 3
    # weights[2] = 3
    # weights /= weights.sum()

    bulletToSMPLLimb = {
        0: 0,
        1: 9,
        2: 15,
        3: 17,
        4: 19,
        5: 21,
        6: 16,
        7: 18,
        8: 20,
        9: 2,
        10: 5,
        11: 8,
        12: 1,
        13: 4,
        14: 7
    }
    if jointsMetric is None:
        jointsMetric = bulletToSMPLLimb.keys()

    dist = 0.

    for ind_BulletLimb in jointsMetric:
        print "Adding limb ", ind_BulletLimb
        ind_SMPLLimb = bulletToSMPLLimb[ind_BulletLimb]
        dist += np.linalg.norm( smpl_joints[ind_SMPLLimb , :] - pybullet[ind_BulletLimb, :] )

    # dist += weights[0] *  np.linalg.norm( smpl_joints[0 , :] - pybullet[0, :] )  # root
    # dist += weights[1] *  np.linalg.norm( smpl_joints[9, :] - pybullet[1, :] )  # chest
    # dist += weights[2] *  np.linalg.norm( smpl_joints[15, :] - pybullet[2, :] )  # neck
    
    # dist += weights[3] *  np.linalg.norm( smpl_joints[17, :] - pybullet[3, :] ) #right shoulder
    # dist += weights[4] *  np.linalg.norm( smpl_joints[19, :] - pybullet[4, :] ) #right elbow
    # dist += weights[5] *  np.linalg.norm( smpl_joints[21, :] - pybullet[5, :] ) #right wrist
    
    # dist += weights[6] *  np.linalg.norm( smpl_joints[16, :] - pybullet[6, :] ) # left arm
    # dist += weights[7] *  np.linalg.norm( smpl_joints[18, :] - pybullet[7, :] )
    # dist += weights[8] *  np.linalg.norm( smpl_joints[20, :] - pybullet[8, :] )
    
    # # dist += weights[9] *  np.linalg.norm( smpl_joints[2, :] - pybullet[9, :] ) # right leg
    # dist += weights[10] *  np.linalg.norm( smpl_joints[5, :] - pybullet[10, :] )
    # dist += weights[11] *  np.linalg.norm( smpl_joints[11, :] - pybullet[11, :] )

    # # dist += weights[12] *  np.linalg.norm( smpl_joints[1, :] - pybullet[12, :] ) # left leg
    # dist += weights[13] *  np.linalg.norm( smpl_joints[4, :] - pybullet[13, :] )
    # dist += weights[14] *  np.linalg.norm( smpl_joints[10, :] - pybullet[14, :] )

    assert not isnan(dist)
    return dist


HUM3D_DESCR = {
    "num_parts": 14,
    "elts_bodypart": [
        "root",
        "root",
        "chest",
        "neck",
        "right hip",
        "right knee",
        "right ankle",
        "right shoulder",
        "right elbow",
        "left hip",
        "left knee",
        "left ankle",
        "left shoulder",
        "left elbow"
    ],
    "elts_type": [
        "position (3D)",
        "rotation (4D)",
        "rotation (4D)",
        "rotation (4D)",
        "rotation (4D)",
        "rotation (1D)",
        "rotation (4D)",
        "rotation (4D)",
        "rotation (1D)",
        "rotation (4D)",
        "rotation (1D)",
        "rotation (4D)",
        "rotation (4D)",
        "rotation (1D)"
    ],
    "metric_points": [
        [0],
        [1, 9, 12],
        [1, 3, 6],
        [2],
        [10], # right hip orientation (knee and ankle)
        [11],
        [],
        [4], # right shoulder
        [5],
        [13], # left hip
        [14],
        [],
        [7], # left shoulder
        [8]
    ],
    "elts_start_ind": [
        1, 4, 8, 12, 16, 20, 21, 25, 29, 30, 34, 35, 39, 43, 44
    ],
    "limbs_opt_order": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
}

def draw_neighboor(x, eps=0.1, ind_bodypart=None):
    x_neigh = np.copy(x)

    # draw index of part which will be perturbed
    if ind_bodypart is None:
        elt_ind = np.random.randint(HUM3D_DESCR["num_parts"])
    else:
        elt_ind = ind_bodypart
    elt_type = HUM3D_DESCR["elts_type"][elt_ind]
    elt_indstart = HUM3D_DESCR["elts_start_ind"][elt_ind]
    elt_indend = HUM3D_DESCR["elts_start_ind"][elt_ind+1]

    # Perturb
    s = slice(elt_indstart, elt_indend)
    if elt_type == "position (3D)":
        x_pert = perturb_position(x[s], eps=eps)
    elif elt_type == "rotation (1D)":
        x_pert = perturb_angle(x[s], eps=3*eps)
    elif elt_type == "rotation (4D)":
        x_pert = perturb_quaternion(x[s], eps=eps)
    else:
        assert False, "Unknown type of joint..."
    
    x_neigh[s] = x_pert
    return x_neigh


def optim_treecoorddescent(x0, joint_3d_position, eps=0.1):
    """
    """
    kmax = 150

    x_k = np.copy(x0)
    cur_dist = distance(x_k, joint_3d_position)

    # first phase: sequential positionning of each limb independently
    for ind_bodypart in HUM3D_DESCR["limbs_opt_order"]:
        k = 0
        cur_dist = distance(x_k, joint_3d_position)

        while (k < kmax):
            print "*** Iteration k:", k
            print " - dist. ", cur_dist
            print " - limb ", HUM3D_DESCR["elts_bodypart"][ind_bodypart]
            print "metric limbs: ", HUM3D_DESCR["metric_points"][ind_bodypart]

            x_cand = draw_neighboor(x_k, ind_bodypart=ind_bodypart, eps=eps)
            cand_dist = distance(x_cand, joint_3d_position, jointsMetric=HUM3D_DESCR["metric_points"][ind_bodypart])

            if (cand_dist < cur_dist):
                x_k = np.copy(x_cand)
                cur_dist = cand_dist

            k += 1

    return x_k



def optim_randcoorddescent(x0, joint_3d_position, eps=0.01):
    """
    """
    kmax = 1000

    x_k = np.copy(x0)
    cur_dist = distance(x_k, joint_3d_position)

    k = 0
    while (k < kmax):
        print "it k:", k
        print "dist. ", cur_dist

        ind_limb = np.random.choice(HUM3D_DESCR["limbs_opt_order"])
        print "limb: ", HUM3D_DESCR["elts_bodypart"][ind_limb]

        x_cand = draw_neighboor(x_k, ind_bodypart=ind_limb, eps=eps)
        cand_dist = distance(x_cand, joint_3d_position)

        if (cand_dist < cur_dist):
            x_k = np.copy(x_cand)
            cur_dist = cand_dist

        k += 1

    return x_k


def optim_coorddescent(x0, joint_3d_position):
    # print joint_3d_position
    for i in range(joint_3d_position.shape[0]):
        for j in range(joint_3d_position.shape[1]):
            print joint_3d_position[i, j]
            print isnan(joint_3d_position[i, j])
            if (isnan(joint_3d_position[i, j])):
                print "--> i, j = ", i, j
                assert False

    k = 0
    # kmax = 50
    x_k = np.copy(x0)
    cur_dist = distance(x_k, joint_3d_position)


    # ## Root positionning
    # # 1 root position (3D),
    # while (k <= 100):
    #     x_cand = np.copy(x_k)
    #     x_cand[1:4] = perturb_position(x_cand[1:4], eps = 0.1)

    #     cand_dist = distance(x_cand, joint_3d_position)

    #     if cand_dist < cur_dist:
            
    #         x_k = np.copy(x_cand)
    #         cur_dist = cand_dist
        

    #     k += 1
    #     print "root  ----------- k ", k
    #     print "cur dist: ", cur_dist
    x_k[1] = -0.04121463
    x_k[2] =  0.96080751
    x_k[3] = -0.02348763


    # ## root orientation
	# # 4 root rotation (4D),
    # k = 0
    # while (k <= 100):
    #     x_cand = np.copy(x_k)
    #     x_cand[4:8] = perturb_quaternion(x_cand[4:8], eps=0.1)

    #     cand_dist = distance(x_cand, joint_3d_position)
    #     if cand_dist < cur_dist:
            
    #         x_k = np.copy(x_cand)
    #         cur_dist = cand_dist

    #     k += 1
    #     print "root  ----------- k ", k
    #     print "cur dist: ", cur_dist
    
    x_k[4] = 0.546853
    x_k[5] =  0.11719007
    x_k[6] = -0.82436401
    x_k[7] = -0.08742007
   

    # ## chest orientation
	# # 8 chest rotation (4D),
    # k = 0
    # while (k <= 20):
    #     x_cand = np.copy(x_k)
    #     x_cand[8:12] = perturb_quaternion(x_cand[8:12], eps=0.1)

    #     cand_dist = distance(x_cand, joint_3d_position)

    #     if cand_dist < cur_dist:
            
    #         x_k = np.copy(x_cand)
    #         cur_dist = cand_dist

    #     k += 1
    #     print "chest ------------ k ", k
    #     print "cur dist: ", cur_dist

    x_k[8] = 0.99676143
    x_k[9] = 0.08016361
    x_k[10] = 0.00526713
    x_k[11] = 0.00356354
    
	# 12 neck rotation (4D),

    ## right hip orientation
	# 16 right hip rotation (4D),
    k = 0
    while (k <= 50):
        x_cand = np.copy(x_k)
        x_cand[16:20] = perturb_quaternion(x_cand[16:20], eps=0.1)

        cand_dist = distance(x_cand, joint_3d_position)

        if cand_dist < cur_dist:
            
            x_k = np.copy(x_cand)
            cur_dist = cand_dist

        k += 1
        print "right hip ------------ k ", k
        print "cur dist: ", cur_dist

    # x_k[16] = 0.99221941
    # x_k[17] = -0.12346259
    # x_k[18] = -0.0074786
    # x_k[19] = 0.01420186
 

    ## right knee orientation
	# 20 right knee rotation (1D),
    k = 0
    while (k <= 50):
        x_cand = np.copy(x_k)
        x_cand[20] = perturb_angle(x_cand[20], eps=0.1)

        cand_dist = distance(x_cand, joint_3d_position)

        if cand_dist < cur_dist:
            
            x_k = np.copy(x_cand)
            cur_dist = cand_dist

        k += 1
        print "right knee ------------ k ", k
        print "cur dist: ", cur_dist

    # x_k[20] = -0.02510488

    ## right ankle orientation
	# 21 right ankle rotation (4D),
    k = 0
    while (k <= 50):
        x_cand = np.copy(x_k)
        x_cand[21:25] = perturb_quaternion(x_cand[21:25], eps=0.1)

        cand_dist = distance(x_cand, joint_3d_position)

        if cand_dist < cur_dist:
            
            x_k = np.copy(x_cand)
            cur_dist = cand_dist

        k += 1
        print "right ankle ------------ k ", k
        print "cur dist: ", cur_dist

    # x_k[21] = 0.74028461
    # x_k[22] = 0.00825344
    # x_k[23] = -0.35407246        
    # x_k[24] = 0.57143965

    # ## right shoulder orientation
	# # 25 right shoulder rotation (4D),
    # k = 0
    # while (k <= 50):
    #     x_cand = np.copy(x_k)
    #     x_cand[25:29] = perturb_quaternion(x_cand[25:29], eps=0.1)

    #     cand_dist = distance(x_cand, joint_3d_position)

    #     if cand_dist < cur_dist:
            
    #         x_k = np.copy(x_cand)
    #         cur_dist = cand_dist

    #     k += 1
    #     print "right shoulder ------------ k ", k
    #     print "cur dist: ", cur_dist

    x_k[25] =  0.98467189
    x_k[26] = -0.14842359
    x_k[27] =  0.00712724
    x_k[28] =  0.09132859


    # ## right elbow orientation
	# # 29 right elbow rotation (1D),
    # k = 0
    # while (k <= 50):
    #     x_cand = np.copy(x_k)
    #     x_cand[29] = perturb_angle(x_cand[29], eps=0.1)

    #     cand_dist = distance(x_cand, joint_3d_position)

    #     if cand_dist < cur_dist:
            
    #         x_k = np.copy(x_cand)
    #         cur_dist = cand_dist

    #     k += 1
    #     print "right elbow ------------ k ", k
    #     print "cur dist: ", cur_dist
    
    x_k[28] =  0.18887186


	# 30 left hip rotation (4D),
	# 31 left knee rotation (1D),
	# 35 left ankle rotation (4D),


    # ## left shoulder orientation
	# # 39 left shoulder rotation (4D),
    # k = 0
    # while (k <= 50):
    #     x_cand = np.copy(x_k)
    #     x_cand[39:43] = perturb_quaternion(x_cand[39:43], eps=0.1)

    #     cand_dist = distance(x_cand, joint_3d_position)

    #     if cand_dist < cur_dist:
            
    #         x_k = np.copy(x_cand)
    #         cur_dist = cand_dist

    #     k += 1
    #     print "left shoulder ------------ k ", k
    #     print "cur dist: ", cur_dist

    x_k[39] = 0.97812104
    x_k[40] =  0.13876954
    x_k[41] = -0.14623018
    x_k[42] =  0.05137105

    # ## left elbow orientation
	# # 43 left elbow rotation (1D)
    # k = 0
    # while (k <= 50):
    #     x_cand = np.copy(x_k)
    #     x_cand[43] = perturb_angle(x_cand[43], eps=0.1)

    #     cand_dist = distance(x_cand, joint_3d_position)

    #     if cand_dist < cur_dist:
            
    #         x_k = np.copy(x_cand)
    #         cur_dist = cand_dist

    #     k += 1
    #     print "left elbow ------------ k ", k
    #     print "cur dist: ", cur_dist

    x_k[43] =  0.07668586



    return x_k