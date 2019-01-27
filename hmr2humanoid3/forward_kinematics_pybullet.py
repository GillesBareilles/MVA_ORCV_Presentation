import pybullet as p
import json
import os
import numpy as np

from utils_pybullet import display_joints_matplotlib

p.connect(p.DIRECT)
#p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP , 1)

import pybullet_data


def getJointPositionsFromHum3d(frameData, target_joints=None):
    """
        Return a 23x3 array of Humanoid3 joint 3d positions at the input pose.

        :param frameData: A Humanoid3 pose description in DeepMimic like format (3d root position
        and relative joint rotations)
    """

    p.resetSimulation()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=200)

    p.loadURDF("plane.urdf",[0,0,-1000])

    humanoid = p.loadURDF("humanoid/humanoid.urdf", globalScaling=0.25)

    for j in range (p.getNumJoints(humanoid)):
        p.changeDynamics(humanoid,j,linearDamping=0, angularDamping=0)

    chest=1
    neck=2
    rightShoulder=3
    rightElbow=4
    leftShoulder=6
    leftElbow = 7
    rightHip = 9
    rightKnee=10
    rightAnkle=11
    leftHip = 12
    leftKnee=13
    leftAnkle=14

    p.getCameraImage(320,200)
    maxForce=1000

    positions = np.zeros((p.getNumJoints(humanoid), 3))

    ## Compute orientations
    basePos1 = [frameData[1],frameData[2],frameData[3]]
    baseOrn1 = [frameData[5],frameData[6], frameData[7],frameData[4]]

    #pre-rotate to make z-up
    y2zPos=[0,0,0.0]
    y2zOrn = p.getQuaternionFromEuler([1.57,0,0])
    basePos,baseOrn = p.multiplyTransforms(y2zPos, y2zOrn, basePos1, baseOrn1)

    chestRot = [frameData[9],frameData[10],frameData[11],frameData[8]]
    neckRot = [frameData[13],frameData[14],frameData[15],frameData[12]]
    
    rightHipRot = [frameData[17],frameData[18],frameData[19],frameData[16]]
    rightKneeRot = [frameData[20]]
    rightAnkleRot = [frameData[22],frameData[23],frameData[24],frameData[21]]
    rightShoulderRot = [frameData[26],frameData[27],frameData[28],frameData[25]]
    rightElbowRot = [frameData[29]]
    
    leftHipRot = [frameData[31],frameData[32],frameData[33],frameData[30]]
    leftKneeRot = [frameData[34]]
    leftAnkleRot = [frameData[36],frameData[37],frameData[38],frameData[35]]        
    leftShoulderRot = [frameData[40],frameData[41],frameData[42],frameData[39]]
    leftElbowRot = [frameData[43]]
    
    p.setGravity(0,0,0)
    

    it = 0
    while (p.isConnected() and it<150):
        it = it + 1
        
        kp=1

        p.resetBasePositionAndOrientation(humanoid, basePos, baseOrn)
        p.setJointMotorControlMultiDof(humanoid, chest, p.POSITION_CONTROL, targetPosition=chestRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, neck, p.POSITION_CONTROL, targetPosition=neckRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, rightHip, p.POSITION_CONTROL, targetPosition=rightHipRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, rightKnee, p.POSITION_CONTROL, targetPosition=rightKneeRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, rightAnkle, p.POSITION_CONTROL, targetPosition=rightAnkleRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, rightShoulder, p.POSITION_CONTROL, targetPosition=rightShoulderRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, rightElbow, p.POSITION_CONTROL, targetPosition=rightElbowRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, leftHip, p.POSITION_CONTROL, targetPosition=leftHipRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, leftKnee, p.POSITION_CONTROL, targetPosition=leftKneeRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, leftAnkle, p.POSITION_CONTROL, targetPosition=leftAnkleRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, leftShoulder, p.POSITION_CONTROL, targetPosition=leftShoulderRot, positionGain=kp, force=maxForce)
        p.setJointMotorControlMultiDof(humanoid, leftElbow, p.POSITION_CONTROL, targetPosition=leftElbowRot, positionGain=kp, force=maxForce)

        if target_joints is not None:
			SMPLLinks = [(0, 2), (2, 5), (5, 8), (8, 11), 
				(0, 1), (1, 4), (4, 7), (7, 10), 
				(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), 
				(9, 14), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
				(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)]

			for (i, j) in SMPLLinks:				
				p.addUserDebugLine(lineFromXYZ=target_joints[i],
								lineToXYZ=target_joints[j],
								lineColorRGB=(0,0,0),
								lineWidth=1,
								lifeTime=0.1)

        p.stepSimulation()


    for j in range (p.getNumJoints(humanoid)):
        res = p.getLinkState(humanoid, j)

        linkWorldPosition = res[0]    # Cartesian position of center of mass

        positions[j, :] = linkWorldPosition

    return positions



if __name__ == "__main__":
    path = os.path.join("/home", "gilles", "Desktop", "ORCV_Project", "DeepMimic", "data", "motions", "humanoid3d_backflip.txt")

    print "\n"
    print "path = ", path
    with open(path, 'r') as f:
        motion_dict = json.load(f)
    #print "motion_dict = ", motion_dict
    print "len motion=", len(motion_dict)
    print motion_dict['Loop']
    numFrames = len(motion_dict['Frames'])
    print "#frames = ", numFrames
    print "\n"

    frame = motion_dict['Frames'][0]

    # getJointPositionsFromHum3d(frame)
    joints_3d_positions = getJointPositionsFromHum3d(frame)
    print joints_3d_positions

    display_joints_matplotlib(joints_3d_positions)