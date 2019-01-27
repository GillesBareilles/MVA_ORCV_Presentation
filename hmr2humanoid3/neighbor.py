import numpy as np
from pyquaternion import Quaternion

def draw_random_angle(eps = np.pi / 8):
    return (np.random.rand() - 0.5) * eps

def perturb_angle(angle, eps = np.pi / 8):
    return angle + draw_random_angle(eps = eps)


def perturb_quaternion(quaternion, eps = 0.1):
    res = np.copy(quaternion)
    pert = (np.random.rand(4) - 0.5) * eps
    res += pert
    res /= np.linalg.norm(res)

    return res

def perturb_position(pos, eps=0.5):
    return pos + (np.random.rand(pos.shape[0]) - 0.5) * eps

def neighboor(s, frame_duration = 0.1):
    snew = np.zeros((44))

    # duration of frame in seconds (1D),
    snew[0] = frame_duration

    # root position (3D),
    snew[1:4] = perturb_position(s[1:4])

    # root rotation (4D),
    snew[4:8] = perturb_quaternion(s[4:8])

    # chest rotation (4D),
    snew[8:12] = perturb_quaternion(s[8:12])

    # neck rotation (4D),
    snew[12:16] = perturb_quaternion(s[12:16])

    # right hip rotation (4D),
    snew[16:20] = perturb_quaternion(s[16:20])

    # right knee rotation (1D),
    snew[20] = perturb_angle(s[20])

    # right ankle rotation (4D),
    snew[21:25] = perturb_quaternion(s[21:25])

    #  right shoulder rotation (4D),
    snew[25:29] = perturb_quaternion(s[25:29])

    # right elbow rotation (1D),
    snew[29] = perturb_angle(s[29])

    # left hip rotation (4D),
    snew[30:34] = perturb_quaternion(s[30:34])
    
    # left knee rotation (1D),
    snew[34] = perturb_angle(s[34])
    
    # left ankle rotation (4D),
    snew[35:39] = perturb_quaternion(s[35:39])

    # left shoulder rotation (4D),
    snew[39:43] = perturb_quaternion(s[39:43])

    # left elbow rotation (1D)
    snew[43] = perturb_angle(s[43])

    print snew

    return snew


if __name__ == "__main__":
    ## Position perturbation
    pos = np.random.rand(3) + 3

    print "-- Init position"
    print pos

    print "\npertubing..."
    print perturb_position(pos)
    print perturb_position(pos)
    print perturb_position(pos)

    ## Angle perturbation
    angle = np.random.rand()

    print "\n-- Init anlge"
    print angle

    print "\npertubing..."
    print perturb_angle(angle)
    print perturb_angle(angle)
    print perturb_angle(angle)

    ## Quaternion perturbation
    quaternion = Quaternion.random()

    print "\n-- Init quaternion"
    print quaternion.norm, quaternion.axis, quaternion.radians

    print "\npertubing..."
    
    q = perturb_quaternion(quaternion)
    print q.norm, q.axis, q.radians
    
    q = perturb_quaternion(quaternion)
    print q.norm, q.axis, q.radians
    
    q = perturb_quaternion(quaternion)
    print q.norm, q.axis, q.radians
