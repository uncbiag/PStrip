from __future__ import print_function

import json
import os
import sys
import numpy as np
import SimpleITK as sitk

sys.path.append(os.path.join(sys.path[0], '..', 'algo_code'))


json_file = os.path.join(sys.path[0], '..', 'config.json')
with open(json_file, 'r') as config_json:
    config = json.load(config_json)

pca_mean = os.path.join(sys.path[0], '..', config['PCA']['pca_mean'])
pca_basis_folder = os.path.join(sys.path[0], '..', config['PCA']['pca_basis_folder'])

num_of_basis = config['PCA']['num_of_basis']
pca_basis_prefix = config['PCA']['pca_basis_prefix']
pca_basis_extension = config['PCA']['pca_basis_extension']


class DecompositionLT(object):

    def __init__(self, input_image, gamma=0.5, platform="GPU", num_of_correction=0):
        self.gamma = gamma
        self.platform = platform
        self.num_of_correction = num_of_correction
        self.input_image = input_image
        self.input_array = sitk.GetArrayFromImage(self.input_image)
        self.lowrank_array = self.input_array
        self.tv_array = np.zeros_like(self.input_array)
        self.basis_array, self.basis_array_transpose, self.mean_array = self.read_pca()

    def __del__(self):
        class_name = self.__class__.__name__
        print(class_name, "destroyed")

    def set_input_array(self, input, input_type='array'):
        if input_type == 'array':
            self.input_array = input
        elif input_type == 'image':
            self.input_array = sitk.GetArrayFromImage(input)
        elif input_type == 'file':
            self.input_array = sitk.GetArrayFromImage(sitk.ReadImage(input))
        return

    def write_to_file(self, lowrank_file, tv_file):
        lowrank_image = sitk.GetImageFromArray(self.lowrank_array)
        tv_image = sitk.GetImageFromArray(self.tv_array)
        lowrank_image.CopyInformation(self.input_image)
        tv_image.CopyInformation(self.input_image)
        sitk.WriteImage(lowrank_image, lowrank_file)
        sitk.WriteImage(tv_image, tv_file)
        return

    def read_pca(self):
        num_of_voxels = self.input_array.size
        basis_array = np.zeros((num_of_voxels, num_of_basis), dtype=np.float32)
        basis_array_transpose = None
        if self.platform == "GPU":
            basis_array_transpose = np.zeros((num_of_basis, num_of_voxels), dtype=np.float32)
        for i in range(num_of_basis):
            pca_image_file = os.path.join(pca_basis_folder, pca_basis_prefix + str(i+1) + pca_basis_extension)
            pca_image_array = sitk.GetArrayFromImage(sitk.ReadImage(pca_image_file))
            basis_array[:,i] = pca_image_array.reshape(-1)
            if self.platform == "GPU":
                basis_array_transpose[i,:] = pca_image_array.reshape(-1)
        mean_array =  sitk.GetArrayFromImage(sitk.ReadImage(pca_mean))
        return basis_array, basis_array_transpose, mean_array

    def decompose(self, verbose):
        dimension = self.input_array.ndim
        D = self.input_array - self.mean_array
        if dimension == 2:
            if self.platform == "GPU":
                print("Dimension 2, GPU")
            else:
                print("Dimension 2, CPU")
        elif dimension == 3:
            if self.platform == "GPU":
                print("Dimension 3, GPU")
                algo = __import__('algo')
                algo.decompose(D, self.basis_array, self.basis_array_transpose, self.gamma, verbose)
            else:
                print("Dimension 3, CPU")
                algo = __import__('algo_cpu')
                algo.decompose(D, self.basis_array, self.gamma, verbose)
        else:
            print ("Wrong Dimension!")

        return


class DecompositionLTS(DecompositionLT):
    def __init__(self, input_image, gamma=0.5, _lambda=0.1, platform="GPU", num_of_correction=0):
        self._lambda = _lambda
        self.atlas_map = config['ATLAS']['atlas_map']
        super(DecompositionLTS, self).__init__(input_image, gamma, platform, num_of_correction)

    def decompose(self):
        print("Start decomposing")
        return

def main():
    image_name = "../../atlas/atlas_wo_skull.nii"
    image = sitk.ReadImage(image_name)
    decomposition = DecompositionLT(image, platform="CPU")
    decomposition.decompose(verbose=True)

if __name__ == '__main__':
    print('This is Decomposition main')
    main()