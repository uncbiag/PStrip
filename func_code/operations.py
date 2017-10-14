import numpy as np
import SimpleITK as sitk
import os

def remove_nan(img_name):
    # remove the nan value inside the img

    img = sitk.ReadImage(img_name)
    img_arr = sitk.GetArrayFromImage(img)
    new_arr = np.nan_to_num(img_arr)
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img, img_name)


def save_image_from_data_matrix(data_matrix, ref_img_name, file_name, extension=False, firstOnly = False):
    ref_img = sitk.ReadImage(ref_img_name)
    ref_arr = sitk.GetArrayFromImage(ref_img)
    shape = ref_arr.shape
    num_of_data = data_matrix.shape[1]
    for i in range(num_of_data):
        img_arr = np.array(data_matrix[:,i].reshape(shape))
        img = sitk.GetImageFromArray(img_arr)
        img.SetOrigin(ref_img.GetOrigin())
        img.SetSpacing(ref_img.GetSpacing())
        img.SetDirection(ref_img.GetDirection())
        if extension != False:
            file_name = file_name + '_' + str(i) + extension
        sitk.WriteImage(img, file_name)
        if firstOnly:
            break

    return

