import sys
import subprocess
import os
import SimpleITK as sitk
import numpy as np

sys.path.append(os.path.join(sys.path[0], '..', 'func_code'))

from niftyreg import *

def create_temp_image(image_path, temp_folder):
    print 'creating temp image'
    temp_image_file = temp_folder + '/temp_image.nii.gz'
    image = sitk.Cast(sitk.ReadImage(image_path), sitk.sitkFloat32)
    temp_image = sitk.RescaleIntensity(image, outputMinimum=1, outputMaximum=1000)
    sitk.WriteImage(temp_image, temp_image_file)
    del image, temp_image 

def affine_to_atlas(atlas_wo_skull_file, atlas_w_skull_file, atlas_mask_file, temp_folder):
    print 'performing affine registration'
    temp_image_file = temp_folder + '/temp_image.nii.gz'
    affine_file1 = temp_folder + '/affine_output1.nii.gz'
    affine_trans1 = temp_folder + '/affine_trans1.txt'
    affine_file2 = temp_folder + '/affine_output2.nii.gz'
    affine_trans2 = temp_folder + '/affine_trans2.txt'

    affine_trans = temp_folder + '/affine_trans.txt'
    invaff_trans = temp_folder + '/affine_invtrans.txt'
    

    affine_log = temp_folder + '/pre_affine.log'
    log = open(affine_log, 'w')

    cmd = ""
    cmd += '\n' + nifty_reg_affine(ref=atlas_w_skull_file, flo=temp_image_file, res=affine_file1, aff=affine_trans1, symmetric=False, rmask=atlas_w_skull_file, init = 'cog')
    cmd += '\n' + nifty_reg_affine(ref=atlas_wo_skull_file, flo=affine_file1, res=affine_file2, aff=affine_trans2, rmask=atlas_mask_file, symmetric=False)
    cmd += '\n' + nifty_reg_transform(comp1=affine_trans2, comp2=affine_trans1, comp3=affine_trans)
    cmd += '\n' + nifty_reg_transform(invAff1=affine_trans, invAff2=invaff_trans)
    
    #cmd += '\n' + Nifty_Reg_Resample(atlas_file, segMovIm, segOutIm, outTrans)
    print cmd
    process = subprocess.Popen(cmd, shell=True, stdout=log)
    process.wait()
    log.close()

def bias_correction(mask_img, temp_folder):
    print 'performing bias correction' 
    affine_file = temp_folder + '/affine_output2.nii.gz'
    bias_file = temp_folder + '/bias_output.nii.gz'
    affine_img = sitk.ReadImage(affine_file)
    bias_img = sitk.N4BiasFieldCorrection(affine_img, sitk.Cast(mask_img, sitk.sitkInt8), convergenceThreshold=0.001)
    sitk.WriteImage(bias_img, bias_file)
    del affine_img, bias_img

def intensity_normalization(mask_img, temp_folder):
    print 'performing intensity normalization'
    bias_file = temp_folder + '/bias_output.nii.gz'
    norm_file = temp_folder + '/norm_output.nii.gz'
    bias_img = sitk.ReadImage(bias_file)
    bias_arr = sitk.GetArrayFromImage(bias_img)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    intensities = bias_arr[np.where(mask_arr==1)]
    i_max = np.percentile(intensities, 99.9)
    i_min = np.percentile(intensities, 0.1)
    print 'i_max:', str(i_max), 'i_min:', i_min
    a = 0.8/(i_max-i_min)
    b = 0.1 - a*i_min

    norm_img_pre = sitk.Add(sitk.Multiply(bias_img, a) ,b)
    norm_img = sitk.Threshold(sitk.Threshold(norm_img_pre, lower=0.0, upper = 100, outsideValue=0.0), lower = 0.0, upper = 1.0, outsideValue=1.0)    
#    norm_img = sitk.IntensityWindowing(bias_img, windowMinimum=i_min, windowMaximum=i_max, outputMinimum=0.0, outputMaximum=1.0)
    sitk.WriteImage(norm_img, norm_file)
    del bias_img, bias_arr, mask_arr, norm_img
    
def histogram_matching(pca_mean_file, temp_folder):
    print 'performing histogram matching'
    norm_file = temp_folder + '/norm_output.nii.gz'
    match_file = temp_folder + '/match_output.nii.gz'
    norm_img = sitk.ReadImage(norm_file)
    pca_mean_img = sitk.ReadImage(pca_mean_file)
    print norm_img.GetSize()
    print pca_mean_img.GetSize()
    match_img = sitk.HistogramMatching(norm_img, pca_mean_img, numberOfHistogramLevels=1000)
    sitk.WriteImage(match_img, match_file)
    del norm_img, pca_mean_img, match_img


def main(argv):

    image_path = sys.argv[1]
    image_file = os.path.basename(image_path)
    image_name = image_file.split('.')[0] 

    root_folder = os.path.join(sys.path[0], '..')

    atlas_w_skull_file = root_folder + '/data/atlas/atlas_w_skull.nii'
    atlas_wo_skull_file = root_folder + '/data/atlas/atlas_wo_skull.nii'
    atlas_mask_file = root_folder + '/data/atlas/atlas_mask.nii' 
    atlas_dilate_mask_file = root_folder + '/data/atlas/atlas_mask_dilate.nii' 

    pca_mean_file = root_folder + '/data/pca/oasis_warped_pca_100/pca_warped_mean_100.nii'
    atlas_mask_img = sitk.ReadImage(atlas_mask_file)
    mask_img = sitk.BinaryErode(atlas_mask_img, 2)
    temp_folder = os.path.join(os.sys.path[0], '..', 'tmp_res', 'temp_'+image_name)
    os.system('mkdir ' + temp_folder)

    create_temp_image(image_path, temp_folder)
    affine_to_atlas(atlas_wo_skull_file, atlas_w_skull_file, atlas_dilate_mask_file, temp_folder)
    bias_correction(mask_img, temp_folder)
    intensity_normalization(mask_img, temp_folder)
    histogram_matching(pca_mean_file, temp_folder) 

    del atlas_mask_img, mask_img


if __name__ == '__main__':
    main(sys.argv[1:])
