import argparse
from train import Model
import os

# Training settings
parser = argparse.ArgumentParser(description='pb maker')

parser.add_argument("--checkpoint_path", type=str, required=True, help='folder of checkpoint(ckpt) path')
parser.add_argument("--output_pb_path", type=str, required=True, help="path where pb will be saved")

os.chdir(os.path.dirname(os.path.realpath(__file__)))

opt = parser.parse_args()

model = Model(opt, None, None, for_pb_maker=True)
