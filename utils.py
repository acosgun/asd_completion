import torch
import random
import torchvision.transforms as transforms
import numpy as np
import cv2
from poissonblending import blend
import random
from PIL import Image
import PIL.ImageOps


def mask_from_file(img_names, shape):
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    for i in range(bsize):
        name = img_names[i]
        temp_str = str(name).replace("train","masks")
        temp_str = temp_str.replace("test","masks")
        temp_index =len(temp_str) - 4
        rand_mask = random.randint(0,2)
        mask_filename = temp_str[:temp_index] + '_mask' + str(rand_mask) + temp_str[temp_index:]
        #print(mask_filename)
        mask_img = Image.open(mask_filename).convert('L')
        mask_img_inverted = PIL.ImageOps.invert(mask_img)
        mask_trans = transforms.ToTensor()
        mask_transformed = mask_trans(mask_img)
        mask[i,:,:,:] = mask_transformed
    return mask

def gray_to_binary(imgs):
    imgs_binary = imgs.copy_(imgs)
    my_index = 0                        
    for row in imgs:
        my_img = transforms.ToPILImage(mode='L')(row)
        bw = my_img.point(lambda x: 0 if x<160 else 255, '1')  #binarization
        trans1 = transforms.ToTensor()
        my_tensor = trans1(bw)
        imgs_binary[my_index,:,:,:] = my_tensor
        my_index = my_index + 1
    return imgs_binary

def gen_input_mask(
    shape, hole_size,
    hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 is provided,
                holes of size (W, H) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes+1)))
        for _ in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y : offset_y + hole_h, offset_x : offset_x + hole_w] = 1.0
    return mask


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin : ymin + h, xmin : xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    names = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index][0], dim=0)
        batch.append(x)
        names.append(dataset[index][1])
    return (torch.cat(batch, dim=0), names)


def blend_images(x, output, mask):
    x = x.clone().cpu()
    #x = torch.cat((x,x,x), dim=1) # convert to 3-channel format
    output = output.clone().cpu()
    #output = torch.cat((output,output,output), dim=1) # convert to 3-channel format
    mask = mask.clone().cpu()
    #mask = torch.cat((mask,mask,mask), dim=1) # convert to 3-channel format
    num_samples = x.shape[0]
    ret = []

    for i in range(num_samples):
        srcimg = transforms.functional.to_pil_image(output[i])
        msk = transforms.functional.to_pil_image(mask[i])

        srcimg_pixels = srcimg.load()
        for x in range(msk.size[0]):
            for y in range(msk.size[1]):
                pix_val = msk.getpixel((x,y))
                if pix_val == 0:
                    srcimg.putpixel((x,y),0)

        out = transforms.functional.to_tensor(srcimg)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
        
    ret = torch.cat(ret, dim=0)
    return ret

def poisson_blend(x, output, mask):
    """
    * inputs:
        - x (torch.Tensor, required)
                Input image tensor of shape (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor from Completion Network of shape (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of shape (N, 1, H, W).
    * returns:
                An image tensor of shape (N, 3, H, W) inpainted
                using poisson image editing method.
    """
    
    x = x.clone().cpu()
    x = torch.cat((x,x,x), dim=1) # convert to 3-channel format
    output = output.clone().cpu()
    output = torch.cat((output,output,output), dim=1) # convert to 3-channel format
    mask = mask.clone().cpu()
    mask = torch.cat((mask,mask,mask), dim=1) # convert to 3-channel format
    num_samples = x.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(x[i])

        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]

        srcimg = transforms.functional.to_pil_image(output[i])
        msk = transforms.functional.to_pil_image(mask[i])
                
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = np.array(msk)[:, :, [2, 1, 0]]
        
        # compute mask's center
        xs, ys = [], []
        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                if msk[i,j,0] == 255:
                    ys.append(i)
                    xs.append(j)
                
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        #out = cv2.seamlessClone(srcimg, msk, msk, center, cv2.NORMAL_CLONE)
        #out = cv2.seamlessClone(msk, srcimg, msk, center, cv2.NORMAL_CLONE)

        """
        #pil_img1 = Image.fromarray(srcimg[:,:,0], 'L')
        #pil_img1.save('mod_src.png')
                
        #pil_img2 = Image.fromarray(dstimg)
        #pil_img3 = Image.fromarray(out)
        #pil_img4 = Image.fromarray(msk)
        
        #pil_img2.save('dst.png')
        #pil_img3.save('out.png')
        #pil_img4.save('msk.png')

        #import sys
        #sys.exit()
        """


        
        #out = out[:, :]
        #out = out[:, :, [2, 1, 0]]
        out = out[:, :, 0]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret


def poisson_blend_old(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network.
        - output (torch.Tensor, required)
                Output tensor of Completion Network.
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network.
    * returns:
                Image tensor inpainted using poisson image editing method.
    """
    num_samples = input.shape[0]
    ret = []

    # convert torch array to numpy array followed by
    # converting 'channel first' format to 'channel last' format.
    input_np = np.transpose(np.copy(input.cpu().numpy()), axes=(0, 2, 3, 1))
    output_np = np.transpose(np.copy(output.cpu().numpy()), axes=(0, 2, 3, 1))
    mask_np = np.transpose(np.copy(mask.cpu().numpy()), axes=(0, 2, 3, 1))

    # apply poisson image editing method for each input/output image and mask.
    for i in range(num_samples):
        inpainted_np = blend(input_np[i], output_np[i], mask_np[i])
        inpainted = torch.from_numpy(np.transpose(inpainted_np, axes=(2, 0, 1)))
        inpainted = torch.unsqueeze(inpainted, dim=0)
        ret.append(inpainted)
    ret = torch.cat(ret, dim=0)
    return ret
