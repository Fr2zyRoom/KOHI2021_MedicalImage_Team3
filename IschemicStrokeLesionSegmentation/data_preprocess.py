import os
from tqdm import tqdm
import argparse

from util.util import *
from mrtools.load_mr import *
from tools.load_nifti import *


class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to data(dwi,adc)')
        parser.add_argument('--maskroot', type=str, help='path to mask')
        parser.add_argument('--savepoint', required=True, help='path to save preprocessed images(.png)')
        
        self.initialized = True
        return parser
    
    
    def gather_options(self):
        if not self.initilized:  # check if it has been initialized
            parser = argparse.ArgumentParser(description='Generate stage1 patches')
            parser = self.initialize(parser)
            self.parser = parser
        return parser.parse_args()
    
    
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

#         # save to the disk
#         expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
#         util.mkdirs(expr_dir)
#         file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
#         with open(file_name, 'wt') as opt_file:
#             opt_file.write(message)
#             opt_file.write('\n')
    
    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

#         # process opt.suffix
#         if opt.suffix:
#             suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
#             opt.name = opt.name + suffix

        self.print_options(opt)
        self.opt = opt
        return self.opt


def select_mr_sequence(folder_dir, sequence='dwi'):
    for seq in os.listdir(folder_dir):
        if seq.lower().endswith(sequence.lower()):
            return os.path.join(folder_dir, seq)
    return None


def run(opt):
    if opt.maskroot is not None:
        mask_path_ls = load_file_path(opt.maskroot, NIFTI_EXTENSION)
        mask_fname_path_dict = {os.path.splitext(os.path.splitext(os.path.basename(mask_path))[0])[0]: mask_path 
                                for mask_path in mask_path_ls}
    else:
        pass
    for case_name in tqdm(sorted(os.listdir(opt.dataroot))):
        case_dir = os.path.join(opt.dataroot, case_name)
        dwi_dir = select_mr_sequence(case_dir, sequence='dwi')
        adc_dir = select_mr_sequence(case_dir, sequence='adc')

        slice2d_dwi_save_path = os.path.join(opt.savepoint, case_name, 'dwi')
        slice2d_adc_save_path = os.path.join(opt.savepoint, case_name, 'adc')
        gen_new_dir(slice2d_dwi_save_path)
        gen_new_dir(slice2d_adc_save_path)
        if opt.maskroot is not None:
            slice2d_mask_save_path = os.path.join(opt.savepoint, case_name, 'mask')
            gen_new_dir(slice2d_mask_save_path)
        else:
            pass
        dwi_dcm_path_ls = load_file_path(dwi_dir, DCM_EXTENSION)
        adc_dcm_path_ls = load_file_path(adc_dir, DCM_EXTENSION)
        if opt.maskroot is not None:
            mask_3d = read_nifti_mask(mask_fname_path_dict[case_name])
            mask_3d = (mask_3d>0).astype(np.uint8) * 255
            
            if len(dwi_dcm_path_ls) != len(mask_3d):
                _, dwi_2dnorm = load_mr_scans(dwi_dcm_path_ls[:len(mask_3d)], norm_mode = '2d')
            else:
                _, dwi_2dnorm = load_mr_scans(dwi_dcm_path_ls, norm_mode = '2d')

            _, adc_2dnorm = load_mr_scans(adc_dcm_path_ls, norm_mode = '2d')

            if len(dwi_2dnorm) == len(adc_2dnorm) == len(mask_3d):
                #print(f'Save! {case_name}')
                resize_and_save_3d(dwi_2dnorm, slice2d_dwi_save_path)
                resize_and_save_3d(adc_2dnorm, slice2d_adc_save_path)
                resize_and_save_3d(mask_3d, slice2d_mask_save_path)
        else:
            _, dwi_2dnorm = load_mr_scans(dwi_dcm_path_ls, norm_mode = '2d')
            _, adc_2dnorm = load_mr_scans(adc_dcm_path_ls, norm_mode = '2d')
            if len(dwi_2dnorm) == len(adc_2dnorm):
                #print(f'Save! {case_name}')
                resize_and_save_3d(dwi_2dnorm, slice2d_dwi_save_path)
                resize_and_save_3d(adc_2dnorm, slice2d_adc_save_path)


if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
