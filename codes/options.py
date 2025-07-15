import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        self.parser.add_argument("--output", required=True, help="experiment save folder")
        self.parser.add_argument('--batch', default=4, type=int)
        self.parser.add_argument('--weights_init', default='crisp_model.pth')
        self.parser.add_argument('--saved_val_features', default='db_features.pth')
        self.parser.add_argument('--gpu', default='0', type=int)

        self.parser.add_argument('--dataroot', default='')
        self.parser.add_argument('--database_dir', default='ref_db')
        self.parser.add_argument('--query_dir_FID', default='val_FID')
        self.parser.add_argument('--query_dir_ShoeCase', default='val_ShoeCase')
        self.parser.add_argument('--save_all_matches', action='store_true', help='test time augmentation')
        self.parser.set_defaults(save_all_matches=False)

        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--feature_dim', default='1,1')
        self.parser.add_argument('--crime_scene_test_freq', default=1, type=int, help='frequency of evaluation on crime scene test images')
        self.parser.add_argument('--seed', default=31934, type=int)
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        opt.input2 = ['print']
        opt.input = ['depth']

        assert (',' in opt.feature_dim)
        opt.feature_dim = tuple(map(int, opt.feature_dim.split(',')))
        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        os.makedirs(opt.output, exist_ok=True)
        file_name = os.path.join(opt.output, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt = opt
        return self.opt