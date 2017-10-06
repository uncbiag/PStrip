import sys
import subprocess
import os
import numpy as np
import SimpleITK as sitk

sys.path.append(os.path.join(sys.path[0], '..', 'func_code'))

from niftyreg import *

def clearFiles(temp_folder):
    lowrank_file = temp_folder + '/Iter6_InvWarpedLowRank.nii.gz'
    sparse_file = temp_folder + '/Iter6_InvWarpedSparse.nii.gz'
    tv_file = temp_folder + '/Iter6_InvTV.nii.gz'

    saved_lr = temp_folder + '/lowrank.nii.gz'
    saved_sparse = temp_folder+ '/sparse.nii.gz'
    saved_tv = temp_folder + '/totalvariation.nii.gz'
   
    os.system('mv ' + lowrank_file + ' ' + saved_lr)
    os.system('mv ' + sparse_file + ' ' + saved_sparse)
    os.system('mv ' + tv_file + ' ' + saved_tv)

    iters_file = temp_folder + '/Iter*'
    os.system('rm ' + iters_file) 


    



def main(argv):
    image_path = sys.argv[1]
    image_file = os.path.basename(image_path)
    image_name = image_file.split('.')[0]

    temp_folder = os.path.join(os.sys.path[0], '..', 'tmp_res', 'temp_' + image_name)
    original_file = image_path
    final_inv_disp = temp_folder + '/final_inv_DVF.nii'
   
     
    final_mask_file = temp_folder + '/Iter6_FinalMask.nii.gz'
    inv_final_mask_file = temp_folder + '/Origin_FinalMask.nii.gz'

    final_brain_file = temp_folder + '/Origin_FinalBrain.nii.gz'

    cmd = ""
    cmd += '\n' + nifty_reg_resample(ref=original_file, flo=final_mask_file, trans=final_inv_disp, res=inv_final_mask_file)
    post_log = temp_folder + '/post_processing.log'
    log = open(post_log, 'w')
    process = subprocess.Popen(cmd, stdout=log, shell=True)
    process.wait()

    final_mask_img = sitk.ReadImage(inv_final_mask_file)
    original_img = sitk.ReadImage(original_file)   
    
    final_mask_brain_img = sitk.Mask(original_img, final_mask_img)
    
    sitk.WriteImage(final_mask_brain_img, final_brain_file)

  
#    clearFiles(temp_folder)

if __name__ == '__main__':
    main(sys.argv[1:])


