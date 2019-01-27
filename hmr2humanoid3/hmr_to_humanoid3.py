import sys, getopt
import pickle as pk, json
import os
import argparse


from forward_kinematics_smpl import getJointPositionsFromSMPL, SMPLPose_to_Hmu3dSpace
from inverse_kinematics_deepmimic import hmroutput_to_joints3d, SMPLcoords_to_rots



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hmrpkl", type=str,
                        help="Location of the hmr pkl file")
    parser.add_argument("outjson", type=str,
                        help="Location where output json file will be written")
    parser.add_argument("-da", "--doall", action="store_true",
                        help="Process all videos", default=False)
    args = parser.parse_args()


    if not args.doall:
        processOneVideo(args.hmrpkl, args.outjson)
    else:
        processVideos(args.hmrpkl, args.outjson)


def processVideos(inputfolder, outputfolder):

    for tool in ['hammer', 'scythe']:
        tool_path = os.path.join(inputfolder, tool)

        for subdir in os.listdir(tool_path):
            if tool in subdir:
                if not os.path.isfile(os.path.join(outputfolder, subdir+".json")):
                    hmr_filepath = os.path.join(tool_path, subdir, "hmr", "hmr.pkl")
                    processOneVideo(hmr_filepath, os.path.join(outputfolder, subdir+".json"))

    
def processOneVideo(pklfile, outjsonfile):
    """
    Turn the HMR pose dict for a sequence of videos into a DeepMimic compatible
    motion reference file
    """

    print "Loading hmr output...", pklfile
    with open(pklfile, 'r') as f:
        hmrdata = pk.load(f)

    # Build SMPL pose for each frame
    print "Building SMPL pose..."
    joint_3d_positions = hmroutput_to_joints3d(hmrdata)
    
    # Build humanoid3 frame descr from SMPL pose
    print "Converting to DeepMimic frames..."
    frames = SMPLcoords_to_rots(joint_3d_positions, 0.1)

    print "\nWriting frames json"
    deepmimicdict = {
        "Loop": "none",
        "Frames": frames.tolist()
    }

    with open(outjsonfile, "w") as f:
        json.dump(deepmimicdict, f)

    print '\nOutput file written at ', outjsonfile
    return

if __name__ == "__main__":
   main()