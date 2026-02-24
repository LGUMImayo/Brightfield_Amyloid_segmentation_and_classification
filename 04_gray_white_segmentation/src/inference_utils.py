import math
import numpy as np

import torch
from skimage import measure

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_weight_matrix(size):
    matrix=np.zeros((size,size))
    max_layer=size//2
    for layer in range(max_layer+1):
        value=(layer+1)/max_layer
        matrix[layer:size-layer,layer:size-layer]=value
    return matrix

def calculate_mask(sub_image,model,transform,step=2,n_channels=3,image_size=520):
    weight_matrix = create_weight_matrix(image_size)
    n_x=math.floor(sub_image.shape[0]/image_size*step)
    n_y=math.floor(sub_image.shape[1]/image_size*step)
    final_mask=np.zeros([n_channels,sub_image.shape[0],sub_image.shape[1]])
    count_mask=np.zeros([n_channels,sub_image.shape[0],sub_image.shape[1]])
    for i in range(int(n_x/2)):
        for j in range(int(n_y/2)):
            patch=sub_image[int(i*image_size/step):int((i+step)*image_size/step),int(j*image_size/step):int((j+step)*image_size/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0).to(device))
            sm=torch.nn.functional.softmax(predictions['out'],dim=1)
            pred_mask=sm[0].detach().cpu().numpy()
            final_mask[:,int(i*image_size/step):int((i+step)*image_size/step),int(j*image_size/step):int((j+step)*image_size/step)]+=pred_mask*weight_matrix
            count_mask[:,int(i*image_size/step):int((i+step)*image_size/step),int(j*image_size/step):int((j+step)*image_size/step)]+=1*weight_matrix

    for i in range(int(n_x/2)):
        for j in range(int(n_y/2)):
            if (i==0)&(j==0):
                patch=sub_image[-image_size:,-image_size:]
            elif (i==0)&(j!=0):
                patch=sub_image[-image_size:,-int(image_size*(j+step)/step):-int(image_size*j/step)]
            elif (i!=0)&(j==0):
                patch=sub_image[-int(image_size*(i+step)/step):-int(image_size*i/step),-image_size:]
            else:
                patch=sub_image[-int(image_size*(i+step)/step):-int(image_size*i/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0).to(device))
            sm=torch.nn.functional.softmax(predictions['out'],dim=1)
            pred_mask=sm[0].detach().cpu().numpy()
            if (i==0)&(j==0):
                final_mask[:,-image_size:,-image_size:]+=pred_mask*weight_matrix
                count_mask[:,-image_size:,-image_size:]+=1*weight_matrix
            elif (i==0)&(j!=0):
                final_mask[:,-image_size:,-int(image_size*(j+step)/step):-int(image_size*j/step)]+=pred_mask*weight_matrix
                count_mask[:,-image_size:,-int(image_size*(j+step)/step):-int(image_size*j/step)]+=1*weight_matrix
            elif (i!=0)&(j==0):
                final_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-image_size:]+=pred_mask*weight_matrix
                count_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-image_size:]+=1*weight_matrix
            else:
                final_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=pred_mask*weight_matrix
                count_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=1*weight_matrix
    

    for i in range(int(n_x/2)):
        for j in range(int(n_y/2)):
            if (i==0):
                patch=sub_image[-image_size:,int(j*image_size/step):int((j+step)*image_size/step)]
            else:
                patch=sub_image[-int(image_size*(i+step)/step):-int(image_size*i/step),int(j*image_size/step):int((j+step)*image_size/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0).to(device))
            sm=torch.nn.functional.softmax(predictions['out'],dim=1)
            pred_mask=sm[0].detach().cpu().numpy()
            if (i==0):
                final_mask[:,-image_size:,int(j*image_size/step):int((j+step)*image_size/step)]+=pred_mask*weight_matrix
                count_mask[:,-image_size:,int(j*image_size/step):int((j+step)*image_size/step)]+=1*weight_matrix
            else:
                final_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),int(j*image_size/step):int((j+step)*image_size/step)]+=pred_mask*weight_matrix
                count_mask[:,-int(image_size*(i+step)/step):-int(image_size*i/step),int(j*image_size/step):int((j+step)*image_size/step)]+=1*weight_matrix

    for i in range(int(n_x/2)):
        for j in range(int(n_y/2)):
            if (j==0):
                patch=sub_image[int(i*image_size/step):int((i+step)*image_size/step),-image_size:]
            else:
                patch=sub_image[int(i*image_size/step):int((i+step)*image_size/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]
            predictions=model(transform(image=patch)['image'].unsqueeze(0).to(device))
            sm=torch.nn.functional.softmax(predictions['out'],dim=1)
            pred_mask=sm[0].detach().cpu().numpy()
            if (j==0):
                final_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-image_size:]+=pred_mask*weight_matrix
                count_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-image_size:]+=1*weight_matrix
            else:
                final_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=pred_mask*weight_matrix
                count_mask[:,int(i*image_size/step):int((i+step)*image_size/step),-int(image_size*(j+step)/step):-int(image_size*j/step)]+=1*weight_matrix
    
    final_mask=final_mask/count_mask
    
    return final_mask