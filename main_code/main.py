import sys
import numpy as np
import SimpleITK as sitk
import subprocess
import os

sys.path.append(os.path.join(sys.path[0], '..', 'func_code'))
from niftyreg import *
from decomposition import *
from operations import *


def performInitialization(argv):
    configure = {}
    num_of_iteration = 6
    configure['num_of_normal_used'] = 100 # currently fixed number of normal images used. 2D:250;3D:80/100
    configure['num_of_pca_basis'] = 50 # currently fixed number of PCA Basis used. 2D:150;3D:50
    configure['num_of_iteration'] = 6 # currently fixed number of iteration. manually change
    configure['num_of_bspline_iteration'] = 3 # currently fixed 3
    
    configure['image_file'] = argv[0] # temp_folder + image_name
    configure['param_1'] = float(argv[1]) # parameter, usually _lambda
    configure['param_2'] = float(argv[2]) # parameter, usually gamma
    configure['num_of_correction'] = int(argv[3]) # number of correction steps performed
    configure['start_iteration'] = 1

    root_folder = os.path.join(sys.path[0], '..')
    result_folder = root_folder + '/tmp_res' + '/temp_' + os.path.basename(configure['image_file']).split('.')[0]
    
    data_folder = os.path.join(root_folder, 'data')
    atlas_folder = os.path.join(data_folder, 'atlas')
    pca_folder = os.path.join(data_folder, 'pca')    
    data_folder_basis = pca_folder + '/pca_' + str(configure['num_of_normal_used'])

    atlas_w_skull_name = atlas_folder + '/atlas_w_skull.nii'
    atlas_im_name = atlas_folder + '/atlas_wo_skull.nii'
    atlas_map_name = atlas_folder + '/skull_map.nii'
    atlas_mask_name = atlas_folder + '/atlas_mask.nii'
    atlas_mask_dilate_name = atlas_folder + '/atlas_mask_dilate.nii'

    configure['result_folder'] = result_folder
    configure['data_folder_basis'] = data_folder_basis
    configure['root_folder'] = root_folder
    configure['atlas_w_skull_name'] = atlas_w_skull_name
    configure['atlas_im_name'] = atlas_im_name
    configure['atlas_map_name'] = atlas_map_name
    configure['atlas_mask_name'] = atlas_mask_name
    configure['atlas_mask_dilate_name'] = atlas_mask_dilate_name
    return configure


def ReadPCABasis(image_size, configure):
    D = np.zeros((image_size, configure['num_of_pca_basis']), dtype=np.float32)
    DT = np.zeros((configure['num_of_pca_basis'], image_size), dtype=np.float32)
    for i in range(configure['num_of_pca_basis']):
        basis_file = configure['data_folder_basis'] + '/eigen_brains_warped/pca_warped_basis_' + str(i+1) + '.nii'
        basis_img = sitk.ReadImage(basis_file)
        basis_img_arr = sitk.GetArrayFromImage(basis_img)
        D[:,i] = basis_img_arr.reshape(-1)
        DT[i,:] = basis_img_arr.reshape(-1)
    mean_img_file = configure['data_folder_basis'] + '/pca_warped_mean_' + str(configure['num_of_normal_used']) + '.nii'
    D_mean = sitk.GetArrayFromImage(sitk.ReadImage(mean_img_file)).astype(np.float32)
    return D, DT, D_mean


def performIteration(configure, D_Basis, D_BasisT, D_mean, image_size):
    current_folder = configure['result_folder']
    start_iteration = configure['start_iteration']
    
    inputIm = configure['result_folder'] + '/match_output.nii.gz'
    outputIm = current_folder + '/Iter1' + '_Input.nii.gz'
    
    os.system('cp ' + inputIm + ' ' + outputIm)
     
    for it in range(configure['num_of_iteration']):
        current_iter = it + 1
        if current_iter < start_iteration:
            continue
        print 'run iteration ' + str(current_iter)
        if current_iter == 1:
            # first iteration, in original space
            performDecomposition(1, current_folder, D_Basis, D_BasisT, D_mean, image_size,configure) 
        else:
            if configure['num_of_iteration'] - current_iter > configure['num_of_bspline_iteration'] - 1:
                # only perform affine registration
                performRegistration(current_iter, current_folder, configure, registration_type = 'affine')
                performDecomposition(current_iter, current_folder, D_Basis, D_BasisT, D_mean, image_size, configure)
            else:
                performRegistration(current_iter, current_folder, configure, registration_type = 'bspline')
                performDecomposition(current_iter, current_folder, D_Basis, D_BasisT, D_mean, image_size, configure)
                performInverse(current_iter, current_folder, configure)
            InverseToIterFirst(current_iter, current_folder, configure)   
    createCompDisp(current_folder, configure) 
    clearUncessaryFiles(current_folder, configure)
 
    return

def createCompDisp(current_folder, configure):
    atlas_im_name = configure['atlas_im_name']
    
    current_comp_disp = current_folder + '/Iter6_DVF_61.nii'
    temp_image = current_folder + '/temp_image.nii.gz'
    affine_txt = current_folder + '/affine_trans.txt'
    affine_def = current_folder + '/affine_DEF.nii'
    affine_disp = current_folder + '/affine_DVF.nii'
    final_disp = current_folder + '/final_DVF.nii'
    final_inv_disp = current_folder + '/final_inv_DVF.nii'
 
    cmd = ""
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,def1=affine_txt, def2=affine_def)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,disp1=affine_def, disp2=affine_disp)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, ref2=atlas_im_name, comp1=current_comp_disp, comp2=affine_disp, comp3=final_disp)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, invNrr1=final_disp, invNrr2=temp_image, invNrr3=final_inv_disp)
    logFile = open('final.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
    process.wait()
    logFile.close()



def clearUncessaryFiles(current_folder, configure):
    deformations = current_folder + '/Iter*.nii'
    affines = current_folder + '/Iter*.txt'
    logs = current_folder + '/Iter*.log'
    tmp_out = current_folder + '/tmp_out.nii'
    invIter3 = current_folder + '/*InvWarpedIter3*'
    os.system('rm ' + deformations)
    os.system('rm ' + affines)
##    os.system('rm ' + logs)
    os.system('rm ' + invIter3)    



def performInverse(current_iter, current_folder, configure):
    prefix = current_folder + '/Iter' + str(current_iter)
    atlas_im_name = configure['atlas_im_name']
    invWarpedLowRankIm = prefix + '_InvWarpedIter3LowRank.nii.gz'
    lowRankIm = prefix + '_LowRank.nii.gz'
    lowRankSIm = prefix + '_LowRankS.nii.gz'

    cmd = ""
    current_disp = prefix + '_DVF_'+str(current_iter)+'3.nii'
    current_inv_disp =  prefix + '_InvDVF_' + str(current_iter) + '3.nii'
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, invNrr1=current_disp, invNrr2=lowRankIm , invNrr3=current_inv_disp)
    cmd += '\n' + nifty_reg_resample(ref=atlas_im_name, flo=lowRankIm, trans=current_inv_disp, res=invWarpedLowRankIm)
        
    logFile = open(prefix + '_data2.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
    process.wait()
    logFile.close()


def InverseToIterFirst(current_iter, current_folder, configure):
    prefix = current_folder + '/Iter' + str(current_iter)
    inverse_disp = prefix + '_inverseDVF_1'+str(current_iter) + '.nii'
    atlas_mask_name = configure['atlas_mask_name']

    lowRankIm = prefix + '_LowRank.nii.gz'
    sparseIm = prefix + '_Sparse.nii.gz'
    totalIm = prefix + '_TV.nii.gz'
    lowrankTVIm = prefix + '_LowRankTV.nii.gz'
    lowrankSIm = prefix + '_LowRankS.nii.gz'
    
    invLowRankIm = prefix + '_InvWarpedLowRank.nii.gz'
    invSparseIm = prefix + '_InvWarpedSparse.nii.gz'
    invTotalIm = prefix + '_InvTV.nii.gz'
    invAtlasMaskIm = prefix + '_InvAtlasMask.nii.gz'
    invLowRankTVIm = prefix + '_InvLowRankTV.nii.gz'
    invLowRankSIm = prefix + '_InvLowRankS.nii.gz'
 
    cmd = ""
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=lowRankIm, trans = inverse_disp, res=invLowRankIm)
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=sparseIm, trans = inverse_disp, res=invSparseIm)
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=totalIm, trans = inverse_disp, res=invTotalIm)
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=lowrankTVIm, trans = inverse_disp, res=invLowRankTVIm)
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=lowrankSIm, trans = inverse_disp, res=invLowRankSIm)
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=atlas_mask_name, trans=inverse_disp, res=invAtlasMaskIm)
 
    num_of_iteration = configure['num_of_iteration']
    if current_iter == num_of_iteration:
        input_file = prefix + '_Input.nii.gz'
        input_image = sitk.ReadImage(input_file)
        input_mask = sitk.Greater(input_image, 0.001)
        
        not_input_mask = sitk.BinaryNot(input_mask)
        con_img = sitk.ConnectedComponent(not_input_mask)
        bin_label_img = sitk.GreaterEqual(con_img, 2)
        new_mask_img = sitk.Or(bin_label_img, input_mask)

        atlas_mask = sitk.ReadImage(atlas_mask_name)
        final_mask = sitk.And(new_mask_img, atlas_mask)
        mask_name = prefix + '_FinalMask.nii.gz'
        sitk.WriteImage(final_mask, mask_name)
        inv_mask_name = prefix + '_InvFinalMask.nii.gz'
        cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=mask_name, trans=inverse_disp, res=inv_mask_name)
    
    logFile = open(prefix+ 'inverse_final_image' + '_data.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
    process.wait()
    logFile.close()
    return 
 
 
def performRegistration(current_iter, current_folder, configure, registration_type = 'bspline'):
    atlas_im_name = configure['atlas_im_name']
    atlas_mask_name = configure['atlas_mask_name']
    atlas_mask_dilate_name = configure['atlas_mask_dilate_name']
    atlas_w_skull_name = configure['atlas_w_skull_name']
    prefix = current_folder + '/' + 'Iter' + str(current_iter) 
    new_input_image = prefix + '_Input.nii.gz'
    current_input_image = current_folder + '/Iter' + str(current_iter-1) + '_Input.nii.gz'
    initial_input_image= current_folder+'/Iter1_Input' + '.nii.gz'
    tmp_out = current_folder + '/tmp_out.nii'
    if configure['num_of_iteration'] - current_iter > configure['num_of_bspline_iteration'] - 2:
        movingIm = current_folder + '/Iter' + str(current_iter-1) + '_LowRankS.nii.gz'
        current_def = prefix + '_DEF_'+str(current_iter)+str(current_iter-1)+'.nii'
        current_disp = prefix + '_DVF_'+str(current_iter)+str(current_iter-1)+'.nii'
        rmask = atlas_mask_dilate_name
    else:
        movingIm = current_folder + '/Iter' + str(current_iter-1) + '_InvWarpedIter3LowRank.nii.gz'
        current_def = prefix + '_DEF_'+str(current_iter)+'3.nii'
        current_disp = prefix + '_DVF_'+str(current_iter)+'3.nii'
        rmask = False
    cmd  = ""
    if registration_type == 'affine':
        outputTransform = prefix + '_Transform.txt' 
        cmd += '\n' + nifty_reg_affine(ref=atlas_im_name, flo=movingIm, aff=outputTransform, symmetric=False, res=tmp_out, rmask=rmask)
    else:
        outputTransform = prefix + '_Transform.nii'
        cmd += '\n' + nifty_reg_bspline(ref=atlas_im_name, flo=movingIm, cpp=outputTransform, res=tmp_out, rmask=rmask)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,def1=outputTransform, def2=current_def)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,disp1=current_def, disp2=current_disp)
    if current_iter > 2 and registration_type == 'affine':
        prefix_prev = current_folder + '/Iter' + str(current_iter-1)
        previous_comp_def =  prefix_prev + '_DEF_'+str(current_iter-1)+'1.nii'
        current_comp_def = prefix + '_DEF_'+str(current_iter)+'1.nii'
        current_comp_disp = prefix + '_DVF_'+str(current_iter)+'1.nii'
        cmd += '\n' + nifty_reg_transform(ref=current_input_image, ref2=initial_input_image, comp1=current_def, comp2=previous_comp_def, comp3=current_comp_def)
    elif current_iter > 2 and registration_type == 'bspline':
        prefix_prev = current_folder + '/Iter3'
        previous_comp_def =  prefix_prev + '_DEF_31.nii'
        current_comp_def = prefix + '_DEF_'+str(current_iter)+'1.nii'
        current_comp_disp = prefix + '_DVF_'+str(current_iter)+'1.nii'
        cmd += '\n' + nifty_reg_transform(ref=current_input_image, ref2=initial_input_image, comp1=current_def, comp2=previous_comp_def, comp3=current_comp_def)
        
    else:
        current_comp_def = current_def
        current_comp_disp = current_disp
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,disp1=current_comp_def,disp2=current_comp_disp)
    inverse_disp = prefix + '_inverseDVF_1'+str(current_iter)+'.nii'
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, invNrr1=current_comp_disp, invNrr2=initial_input_image, invNrr3=inverse_disp)
    cmd += '\n' + nifty_reg_resample(ref=atlas_im_name,flo=initial_input_image,trans=current_comp_def,res=new_input_image)
    logFile = open(prefix + '_data.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
    process.wait()
    logFile.close()
    return 
             

        
def performDecomposition(current_iter, current_folder, Beta, BetaT, D_mean, image_size, configure):
    # D_mean: image_matrix (x,y,z) or (x,y)
    num_of_iteration = configure['num_of_iteration']
    num_of_bspline_iter = configure['num_of_bspline_iteration']

    _lambda = configure['param_1']
    _gamma = configure['param_2']
    atlas_im_name = configure['atlas_im_name']
    atlas_map = sitk.GetArrayFromImage(sitk.ReadImage(configure['atlas_map_name'])).astype(np.float32)
    correction = configure['num_of_correction']
    
    input_name = current_folder + '/' + 'Iter' + str(current_iter) + '_Input.nii.gz'
    D = sitk.GetArrayFromImage(sitk.ReadImage(input_name)).astype(np.float32)
    if current_iter <= num_of_iteration - 1 and correction != 0:
        L, S, T, Alpha = pca_GPU(D, D_mean, atlas_map, Beta, BetaT, _lambda, _gamma/2, 0)
    else:
        L, S, T, Alpha = pca_GPU(D, D_mean, atlas_map, Beta, BetaT, _lambda, _gamma, correction)
 

    l_v = L.reshape(image_size, 1) # quasi low-rank/reconstruction
    s_v = S.reshape(image_size, 1) # sparse
    t_v = T.reshape(image_size, 1) # total variation term
    lt_v = l_v + t_v
    ls_v = l_v + s_v

    prefix = current_folder + '/' + 'Iter' + str(current_iter)
 
    lowRankIm = prefix + '_LowRank.nii.gz'
    sparseIm = prefix + '_Sparse.nii.gz'
    totalIm = prefix + '_TV.nii.gz'
    lowrankTVIm = prefix + '_LowRankTV.nii.gz'
    lowrankSIm = prefix + '_LowRankS.nii.gz'

    save_image_from_data_matrix(l_v,atlas_im_name,lowRankIm)
    save_image_from_data_matrix(s_v,atlas_im_name,sparseIm)
    save_image_from_data_matrix(t_v,atlas_im_name,totalIm)
    save_image_from_data_matrix(lt_v, atlas_im_name, lowrankTVIm)
    save_image_from_data_matrix(ls_v, atlas_im_name, lowrankSIm)

    return


def main(argv):
    
    configure = performInitialization(argv)
    atlas_map = sitk.GetArrayFromImage(sitk.ReadImage(configure['atlas_map_name']))
    atlas_arr = sitk.GetArrayFromImage(sitk.ReadImage(configure['atlas_im_name']))
    z,x,y = atlas_arr.shape
    image_size = x*y*z

    D_Basis, D_BasisT, D_mean = ReadPCABasis(image_size, configure)
    performIteration(configure, D_Basis, D_BasisT, D_mean, image_size)

if __name__ == '__main__':
    main(sys.argv[1:])
 

