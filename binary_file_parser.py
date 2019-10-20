# Copyright 2017 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from struct import unpack
import numpy as np
import cairocffi as cairo
import sys

def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)
    
    return raster_images

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break

count = 0
for drawing in unpack_drawings('horse.bin'):
    
    if not drawing['recognized'] or not drawing['countrycode'] == b'US':
        continue
    
    from PIL import Image, ImageDraw
    img = Image.new(mode= 'L', size = (256, 256), color=(255))
    img_draw = ImageDraw.Draw(img)

    #init masks
    num_masks = 3
    mask_imgs = []
    mask_imgs_draw = []
    
    for i in range(num_masks):
        mask_img = Image.new(mode= 'L', size = (256, 256), color=(255))
        mask_imgs.append(mask_img)
        mask_imgs_draw.append(ImageDraw.Draw(mask_img))

    num_strokes = len(drawing['image'])
    if num_strokes < 3:
        continue
              
    denom = num_strokes // 3
    rem = num_strokes % 3    
    num_first_last = denom
    num_mid = denom
    if rem == 1:
        num_mid += 1
    elif rem == 2:
        num_first_last += 1
    mask_indices = [[1]*num_first_last + [0]*num_mid + [0]*num_first_last,
                    [0]*num_first_last + [1]*num_mid + [0]*num_first_last,
                    [0]*num_first_last + [0]*num_mid + [1]*num_first_last]    

    #print("n_strokes: " + str(num_strokes))
    #print("[" + str(num_first_last) +","+ str(num_mid) + "," + str(num_first_last) + "]")
    
    stroke_index = 0
    for stroke in drawing['image']:
        
        stroke_len = len(stroke[0])
        for i in range(0, stroke_len-1):
            x1 = stroke[0][i]
            y1 = stroke[1][i]
            x2 = stroke[0][i+1]
            y2 = stroke[1][i+1]
            img_draw.line((x1,y1,x2,y2), fill=(0), width=8)

            for m_ind in range(num_masks):                
                if mask_indices[m_ind][stroke_index] == 0:
                    mask_imgs_draw[m_ind].line((x1,y1,x2,y2), fill=(0), width=8)
            
        stroke_index += 1
            
    img = img.resize((64,64),Image.NEAREST)
    img_name = "horse" + ("%06d" % count) + ".png"
    img.save("./horses/" + img_name, "PNG")
    
    for mask_ind, mask_img in enumerate(mask_imgs):
        mask_img = mask_img.resize((64,64),Image.NEAREST)
        mask_img_name = "horse" + ("%06d" % count) + "_mask" + str(mask_ind) + ".png"
        mask_img.save("./horses/masks/" + mask_img_name, "PNG")
    
    count = count + 1
    #sys.exit()
