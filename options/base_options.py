#a few codes come from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import argparse

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing.
    #It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--dataroot', default='/home/limingchao/PycharmProjects/untitled/Dataset/BJ_Dataset/Dataset', help='path to data')
        parser.add_argument('--Network_mode', type=str, default='ST', help='S,T,ST')
        parser.add_argument('--backbone', type=str, default='VGG', help='VGG,ResNet16')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids')
        parser.add_argument('--input_size', type=list, default=[512,768,3], help='input data size separated with comma')
        parser.add_argument('--in_channels', type=int, default=3, help='input channels')
        parser.add_argument('--channels', type=int, default=64, help='channels')
        parser.add_argument('--saveroot', default='logs', help='path to save results')
        parser.add_argument('--n_classes', type=int, default=4, help='final class number for classification')
        parser.add_argument('--print_cam', type=bool, default=False, help='print grad_cam')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
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
        print('')

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt



