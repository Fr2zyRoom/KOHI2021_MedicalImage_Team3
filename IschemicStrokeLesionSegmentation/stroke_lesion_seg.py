import argparse
from tqdm import tqdm
from util.util import * 
from dataset.load_data import *
from dataset.seg_dataset import *
import segmentation_models_pytorch as smp


class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to data(dwi,adc)')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--savepoint', required=True, help='path to save preprocessed images(.png)')
        parser.add_argument('--weights', required=True, help='pre-trained weights')
        
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


def run(opt):
    test_dataset = DWI_ADC_IschemicStrokeLesionSegDataset(
        img_folder_dir = opt.dataroot, 
        augmentation=None, 
        preprocessing=get_preprocessing(resize=(256,256))
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # load best saved checkpoint
    model = torch.load(opt.weights)
    
    print('Inference...')
    
    DEVICE='cuda'
    predict_masks = []

    for data in tqdm(test_loader):
        images = data
        images = images.to(DEVICE)
        pr_mask = model.predict(images)
        predict_masks.append(pr_mask.cpu().numpy())
    
    predict_masks = np.squeeze(np.vstack(predict_masks))
    predict_masks_norm = (predict_masks*255).astype(np.uint8)
    
    print('Save results...')
    
    len_cnt = 0
    for fname in tqdm(test_dataset.fname_list):
        case_len = len([p for p, _ in test_dataset.img_path_arr if fname in p.split(os.path.sep)[-3]])
        tmp = predict_masks_norm[len_cnt:len_cnt+case_len]
        savepoint = os.path.join(opt.savepoint, fname, 'pred_masks')
        gen_new_dir(savepoint)
        for i, slice_img in enumerate(tmp):
            save_name = str(i).zfill(3)
            save_arr_to_png(slice_img, savepoint, save_name)
        len_cnt += case_len
    
    print('Done!')
    

if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
