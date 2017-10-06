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
    if os.path.isfile(temp_image_file):
        return
    # using float32
    image = sitk.Cast(sitk.ReadImage(image_path), sitk.sitkFloat32)
    temp_image = sitk.RescaleIntensity(image1=image, outputMinimum=0, outputMaximum=1)
    sitk.WriteImage(temp_image, temp_image_file)
    del image 

def affine_to_atlas(atlas_wo_skull_file, atlas_w_skull_file, atlas_mask_file, temp_folder):

    # Two steps of affine registration
    print 'performing affine registration'
    # input image 
    temp_image_file = temp_folder + '/temp_image.nii.gz'
    # affine output/trans 1
    affine_file1 = temp_folder + '/affine_output1.nii.gz'
    affine_trans1 = temp_folder + '/affine_trans1.txt'
    # affine output/trans 2
    affine_file2 = temp_folder + '/affine_output2.nii.gz'
    affine_trans2 = temp_folder + '/affine_trans2.txt'
    # affine trans/inv_trans
    affine_trans = temp_folder + '/affine_trans.txt'
    invaff_trans = temp_folder + '/affine_invtrans.txt'

    if os.path.isfile(invaff_trans):
        return
    affine_log = temp_folder + '/pre_affine.log'
    log = open(affine_log, 'w')
    cmd = ""
    # reg input -> atlas_w_skull, mask on atlas_w_skull
    cmd += '\n' + nifty_reg_affine(ref=atlas_w_skull_file, flo=temp_image_file, res=affine_file1, aff=affine_trans1, symmetric=False, init = 'cog')
    # reg affine output 1 -> atlas_wo_skull, mask on dilate
    cmd += '\n' + nifty_reg_affine(ref=atlas_wo_skull_file, flo=affine_file1, res=affine_file2, aff=affine_trans2, rmask=atlas_mask_file, symmetric=False)
    # composite trans
    cmd += '\n' + nifty_reg_transform(comp1=affine_trans2, comp2=affine_trans1, comp3=affine_trans)
    # get inverse
    cmd += '\n' + nifty_reg_transform(invAff1=affine_trans, invAff2=invaff_trans)

    print cmd
    process = subprocess.Popen(cmd, shell=True, stdout=log)
    process.wait()
    log.close()


def bias_correction(atlas_erode_mask_file, temp_folder):
    print 'performing bias correction' 

    # input/output file name
    affine_file = temp_folder + '/affine_output2.nii.gz'
    bias_file = temp_folder + '/bias_output.nii.gz'
    if os.path.isfile(bias_file):
        return
     
    # input image
    affine_img = sitk.ReadImage(affine_file)
    
    # rescale input image to [1, 1000], just to remove small and negative intensity
    rescale_affine_img = sitk.RescaleIntensity(image1=affine_img, outputMinimum=1, outputMaximum=1000)
    mask_img = sitk.ReadImage(atlas_erode_mask_file)
    bias_img = sitk.N4BiasFieldCorrection(image=rescale_affine_img, maskImage=mask_img)

    sitk.WriteImage(bias_img, bias_file)
    del affine_img, bias_img, rescale_affine_img

def intensity_normalization(atlas_erode_mask_file, temp_folder):
    print 'performing intensity normalization'
    
    # input/output file name
    bias_file = temp_folder + '/bias_output.nii.gz'
    norm_file = temp_folder + '/norm_output.nii.gz'
    
    if os.path.isfile(norm_file):
        return
    bias_img = sitk.ReadImage(bias_file)
    bias_arr = sitk.GetArrayFromImage(bias_img)
    mask_img = sitk.ReadImage(atlas_erode_mask_file)
    mask_arr = sitk.GetArrayFromImage(mask_img)

    # calculate 99th and 1st percentile
    intensities = bias_arr[np.where(mask_arr==1)]
    i_max = np.percentile(intensities, 99)
    i_min = np.percentile(intensities, 1)
    print 'i_max:', str(i_max), 'i_min:', i_min
    # map i_max -> 0.9, i_min -> 0.1, affine tranform on intensities, then cutoff [0, 1]
    # y = a(x+b)
    b = (i_max-9*i_min)/8
    a = 0.8/(i_max-i_min)
    norm_img_pre = sitk.ShiftScale(image1=bias_img, shift=b, scale=a)
#    a = 0.8/(i_max-i_min)
#    b = 0.1 - a*i_min
#    norm_img_pre = sitk.Add(sitk.Multiply(bias_img, a) ,b)

    # cutoff at 0, 1
    norm_img = sitk.IntensityWindowing(norm_img_pre, windowMinimum=0.0, windowMaximum=1.0, outputMinimum=0.0, outputMaximum=1.0)
#    norm_img = sitk.Threshold(sitk.Threshold(norm_img_pre, lower=0.0, upper = 100, outsideValue=0.0), lower = 0.0, upper = 1.0, outsideValue=1.0)    

    sitk.WriteImage(norm_img, norm_file)
    del bias_img, bias_arr, mask_arr, mask_img, norm_img_pre, norm_img
    
def histogram_matching(pca_mean_file, atlas_erode_mask_file, temp_folder):
    print 'performing histogram matching'
    # input/output image
    norm_file = temp_folder + '/norm_output.nii.gz'
    match_file = temp_folder + '/match_output.nii.gz'
    if os.path.isfile(match_file):
        return
    
    norm_img = sitk.ReadImage(norm_file)
    mask_img = sitk.ReadImage(atlas_erode_mask_file)   
    
    # only match inside of erode mask
    norm_in_img = sitk.Mask(image=norm_img, maskImage=mask_img)
    norm_out_img = sitk.MaskNegated(image=norm_img, maskImage=sitk.Cast(mask_img, norm_img.GetPixelID()))
    pca_mean_img = sitk.ReadImage(pca_mean_file)
 
    match_in_img = sitk.HistogramMatching(norm_in_img, pca_mean_img, numberOfHistogramLevels=1000)
    match_img = sitk.Add(match_in_img, norm_out_img)
    sitk.WriteImage(match_img, match_file)
    del norm_img, mask_img, norm_in_img, norm_out_img, pca_mean_img, match_in_img, match_img



def main(argv):
    image_path = sys.argv[1]
    image_file = os.path.basename(image_path)
    image_name = image_file.split('.')[0] 

    root_folder = os.path.join(sys.path[0], '..')

    atlas_w_skull_file = root_folder + '/data/atlas/atlas_w_skull.nii'
    atlas_wo_skull_file = root_folder + '/data/atlas/atlas_wo_skull.nii'
    atlas_mask_file = root_folder + '/data/atlas/atlas_mask.nii' 
    atlas_dilate_mask_file = root_folder + '/data/atlas/atlas_mask_dilate.nii' 
    atlas_erode_mask_file = root_folder + '/data/atlas/atlas_mask_erode.nii'
    pca_mean_file = root_folder + '/data/pca/pca_100/pca_warped_mean_100.nii'

    temp_folder = os.path.join(os.sys.path[0], '..', 'tmp_res', 'temp_'+image_name)
    os.system('mkdir ' + temp_folder)

    create_temp_image(image_path, temp_folder)

    # using dilate mask file for second affine
    affine_to_atlas(atlas_wo_skull_file, atlas_w_skull_file, atlas_dilate_mask_file, temp_folder)

    bias_correction(atlas_erode_mask_file, temp_folder)
    intensity_normalization(atlas_erode_mask_file, temp_folder)
    histogram_matching(pca_mean_file, atlas_erode_mask_file, temp_folder) 

if __name__ == '__main__':
    main(sys.argv[1:])
