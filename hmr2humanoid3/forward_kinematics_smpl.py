import numpy as np
import numpy.linalg as LA
import chumpy as ch
import os.path as OP
from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation

from pyquaternion import Quaternion

def getJointPositionsFromSMPL(beta, theta, gender='male', n_betas=10):
    '''                                                                                              
    Computes 3D joint positions from SMPL shape vector, beta, and SMPL pose vector, theta.           
    theta: a SMPL pose vector of dimension 72                                                        
    beta: a SMPL shape vector of dimension 10                                                        
    '''
    # load SMPL body model
    model_dir = '/home/gilles/Desktop/ORCV_Project/smpl/models'

    # load model
    if gender == 'neutral':
        model_path = OP.join(model_dir, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    elif gender == 'male':
        model_path = OP.join(model_dir, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    elif gender == 'female':
        model_path = OP.join(model_dir, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    model = load_model(model_path)

    # get the joint 3D positions of a SMPL body in canonical pose
    j3d_dirs = np.dstack([model.J_regressor.dot(model.shapedirs[:,:,i]) for i in range(n_betas)])
    j3d_canonical = ch.array(j3d_dirs).dot(beta) + model.J_regressor.dot(model.v_template.r)
    
    # get joint 3D positionss via forward kinematics
    (_, A_global) = global_rigid_transformation(theta, j3d_canonical, model.kintree_table, xp=ch)
    joint_3d_positions = ch.vstack([g[:3, 3] for g in A_global]).r
    return joint_3d_positions

def SMPLPose_to_Hmu3dSpace(joint_3d_positions):
    joint_3d_normalized = np.zeros(joint_3d_positions.shape)

    ## Rotation
    q_rot = Quaternion(axis=[1, 0, 0], radians=-np.pi / 2)

    for ind, vec3 in enumerate(joint_3d_positions):
        joint_3d_normalized[ind, :] = q_rot.rotate(vec3)


    ## Translation s.t. feets touch fround
    minz = joint_3d_normalized[:, 2].min()
    translation = np.array([0, 0, -minz])

    for ind, vec3 in enumerate(joint_3d_positions):
        joint_3d_normalized[ind, :] += translation

    return joint_3d_normalized  