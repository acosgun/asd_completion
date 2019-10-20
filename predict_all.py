import os
import argparse
import torch
import json
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import CompletionNetwork
from PIL import Image
import PIL.ImageOps
from utils import poisson_blend, gen_input_mask, gray_to_binary



parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)

torch.set_printoptions(threshold=5000)

def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_img = os.path.expanduser(args.input_img)
    args.output_img = os.path.expanduser(args.output_img)


    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1,1,1,1)
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))


    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    import torchvision as tv
    img = Image.open(args.input_img)
    #img = tv.transforms.Grayscale(num_output_channels=1),
    img = transforms.Resize(args.img_size)(img)
    img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    # create mask
    mask = gen_input_mask(
        shape=(1, 1, x.shape[2], x.shape[3]),
        hole_size=(
            (args.hole_min_w, args.hole_max_w),
            (args.hole_min_h, args.hole_max_h),
        ),
        max_holes=args.max_holes,
    )
    #print(mask.shape)
    #print(mask)
    temp_str = str(args.input_img).replace("test","masks")
    temp_index =len(temp_str)-4


    out_img = torch.Tensor()
    for i in range(3):
        #print(mask_filename)
        mask_filename = temp_str[:temp_index] + '_mask' + str(i) + temp_str[temp_index:]        
        mask_img = Image.open(mask_filename).convert('L')
        mask_img_inverted = PIL.ImageOps.invert(mask_img)
        #mask_transformed = mask_trans(mask_img_inverted)
        
        mask_trans = transforms.ToTensor()
        mask_transformed = mask_trans(mask_img)
        
        mask_shape=(1, 1, x.shape[2], x.shape[3])
        new_mask = torch.zeros(mask_shape)
        new_mask[0,0,:,:] = mask_transformed
        mask = new_mask

        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask
            input = torch.cat((x_mask, mask), dim=1)
            output = model(input)
            inpainted = poisson_blend(x, output, mask)
            binary_out = inpainted.clone()
            binary_out = gray_to_binary(binary_out)
            out_img = torch.cat((out_img, x, x_mask, inpainted, binary_out), dim=0)
    save_image(out_img, args.output_img, nrow=4)
        
    print('output img was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
