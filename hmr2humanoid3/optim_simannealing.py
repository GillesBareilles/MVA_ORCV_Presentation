# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


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


def P(delta_ener, temp):
    return np.exp( - delta_ener / temp)

def sim_annealing(s0, target, logging=True):
    H_temps = []
    H_score = []
    H_temp = []

    T0 = 10.0
    Tmin = 1e-2
    tau = 1e4

    kmax = 5000
    T = T0

    s = np.copy(s0)
    e = distance(s0, target)
    emin = e
    smin = np.copy(s)
    k = 0
    while ((k < kmax) and (T > Tmin)):
        print "-- iteration ", k
        print "- s: ", s
        print "- e: ", e
        print "- T: ", T

        sn = neighboor(s)
        en = distance(sn, target)

        while isnan(en):
            print sn
            assert False
            # sn = neighboor(s)
            # en = distance(sn, target)

        T = T0*np.exp(-k/tau)

        # if ((en < e) or (np.random.rand() < P(en - e, T))):
        if ((en < e)):
        # si en < e ou aléatoire() < P(en - e, temp(k/kmax)) alors
            if (en < e):
                print "Improvement !"
            else:
                print "Degrading by ", en - e

            s = np.copy(sn)
            e = en
        
        if (e < emin):
            print "New best position ", emin
            emin = e
            smin = np.copy(s)

        if (logging and (k % 1 == 0)):
            H_temps.append(k)
            H_score.append(e)
            H_temp.append(T)
        k = k + 1

    if logging:
        plt.figure()
        plt.subplot(1,2,1)
        plt.semilogy(H_temps, H_score)
        plt.title("Evolution de l'energie totale du systeme")
        plt.xlabel('Temps')
        plt.ylabel('Energie')
        plt.subplot(1,2,2)
        plt.semilogy(H_temps, H_temp)
        plt.title('Evolution de la temperature du systeme')
        plt.xlabel('Temps')
        plt.ylabel('Temperature')
        plt.show()
            
    return smin
