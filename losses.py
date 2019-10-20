from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms
from PIL import Image
import sys
import torch.nn as nn

def completion_network_loss(input, output, mask):

    #input = input.masked_fill(input >= 0.5,1.0)
    #input = input.masked_fill(input <  0.5,0.0)
    
    input = input[:,0,:,:]
    output = output[:,0,:,:]
    mask = mask[:,0,:,:]
    
    #output = output.masked_fill(output >= 0.5,1.0)
    #output = output.masked_fill(output <  0.5,0.0)
    
    #input_img = transforms.functional.to_pil_image(input[0,:,:].cpu()*mask[0,:,:].cpu())
    #output_img = transforms.functional.to_pil_image(output[0,:,:].cpu()*mask[0,:,:].cpu())

    
    #input_img.show()
    #output_img.show()
    
    #criterion = CrossEntropyLoss().cuda()
    #sys.exit()

    #total_loss = 0    
    #for i in range(input.shape[0]):
    #    total_loss = total_loss + criterion(output[i,:,:] * mask[i,:,:], (input[i,:,:] * mask[i,:,:]).long())
    criterion = nn.BCELoss()
    return criterion(output * mask, input)
    #return criterion(output, input)

    #return mse_loss(output * mask, input * mask)
    #return mse_loss(output, input)
    #return criterion(output, input)
