import argparse
import os
import warnings

from preprocessing import *
from main import *
from postprocessing import *


def is_file_exist(file_fullpath):
#   check if image exists
    if not os.path.isfile(file_fullpath):
        msg = 'The input image file does not exist!'
        raise argparse.ArgumentTypeError(msg)

#    base_name = os.path.basename(file_fullpath)
#    image_name = base_name.split('.')[0]
#    extension = '.'.join(base_name.split('.')[1:])
#   check file extension  (FUTURE)
#    if not extension in ['nii.gz', 'nii', 'hdr', 'img']:
#        msg = 'The input image file extension is not supported!'
#        raise argparse.ArgumentTypeError(msg)
    return file_fullpath

parser = argparse.ArgumentParser(description='PCA model for Brain Extraction: We provide arguments for user to specify, although the default settings are sufficient for most of the cases.', prog='PStrip', conflict_handler='resolve', usage='python pstrip.py -i Input_Image [-m Mask_Name] [-o Output_Name] [-p platform] [-g gamma] [-l lambda] [-c correction] [-v] [-h]')
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-i', '--input', metavar='', dest='input_image', help='input image name', required=True, type=is_file_exist)

file_args = parser.add_argument_group('optional file arguments')
file_args.add_argument('-m', '--mask', metavar='', dest='mask_image', help='output mask name. If both mask and output image are not specified, mask image will be generated into input image folder.')
file_args.add_argument('-o', '--output', metavar='', dest='output_image', help='brain extracted image. If both mask and output image are not specified, output image will be generated into input image folder.')
#file_args.add_argument('-l', '--lowrank', metavar='', dest='lowrank_image', help='quasi-normal image [quasi-lowrank]. If not specified, the image will not be generated.')
#file_args.add_argument('-s', '--sparse', metavar='', dest='sparse_image', help='sparse image. If not specified, the image will not be generated.')
#file_args.add_argument('-t', '--tv', metavar='', dest='tv_image', help='tv image. If not specified, the image will not be generated.')





additional_args = parser.add_argument_group('additional arguments')
additional_args.add_argument('-p', '--platform', metavar='', dest='platform', help='platform (CPU/GPU), default GPU', default='GPU', choices=['CPU', 'GPU', 'cpu', 'gpu'])
additional_args.add_argument('-g', '--gamma', metavar='', dest='gamma', help='gamma for total variation term penalty, default 0.5', type=float, default=0.5)
additional_args.add_argument('-la', '--lambda', metavar='', dest='_lambda', help='lambda for sparse term penalty, default 0.1', type=float, default=0.1)
additional_args.add_argument('-c', '--correction', metavar='', dest='num_of_correction', help='number of correction (regularization) steps, default 0', type=int, default=0)
additional_args.add_argument('-d', '--debug', metavar='', dest='debug', help='Debug mode:\n 0: no intermediate results will be saved. \n 1: some of itermediate results will be saved. \n 2: all intermediate results will be saved in tmp_res folder. Be careful, this could occupy large space on disk if multiply images will be processed', type=int, choices=[0,1,2], default=0)
additional_args.add_argument('-v', '--version', action='version', version='PStrip v1.0: PCA model for Brain Extraction')
additional_args.add_argument('-h', '--help', action='help', help='show this help message and exit')
additional_args.add_argument('-V', '--verbose', action='store_true', help='increase output verbosity')
args = parser.parse_args()

if __name__ == '__main__':

    if args.mask_image == None and args.output_image == None: 
        msg = 'No output image and mask image specified. Output images will be stored in image folder'
        warnings.warn(message=msg, category=Warning)

        input_image = args.input_image   #/path/to/image/image_name.extension
        dir_name = os.path.dirname(input_image) #/path/to/image/
        base_name = os.path.basename(input_image) #image_name.extension
        image_name = base_name.split('.')[0] #image_name
        extension = '.'.join(base_name.split('.')[1:]) #extension

        mask_name = image_name + '_mask' + '.' + extension 
        args.mask_image = os.path.join(dir_name, mask_name) #/path/to/image/image_name_mask.extension
        output_name = image_name + '_extracted' + '.' + extension
        args.output_image = os.path.join(dir_name, output_name) #/path/to/image/image_name_extracted.extension
 
    if args.debug == 2:
        msg = 'You are using ultra debugging mode. All intermediate results will be saved on disk. This could occupy large space on disk if multiple images will be processed. Consider use -d 1.'
        warnings.warn(message=msg, category=Warning)

    args.platform = args.platform.upper()
    if args.verbose == True:
        print 'input parameters: ' + str(args)

    print 'Starting pre-processing'   
    preprocessing(args)


    print 'Starting Decomposition/Registration'
    main(args)
    print 'Starting post-processing'
    postprocessing(args)

