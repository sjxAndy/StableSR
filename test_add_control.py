import sys
import os

# assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
# output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from omegaconf import OmegaConf
# from share import *
# from cldm.model import create_model
from ldm.util import log_txt_as_img, exists, instantiate_from_config



def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


# model = create_model(config_path='./models/cldm_v15.yaml')
config = OmegaConf.load('/home/sjx22/code/StableSR/configs/stableSRNew/v2-finetune_face_T_512_ref_stablesr_control.yaml') 
model = instantiate_from_config(config['model'])
model2 = instantiate_from_config(config['model'])

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']


pre2 = torch.load('/home/sjx22/code/StableSR/logs/2024-07-10T04-37-20_train_0710_0/checkpoints/last.ckpt', map_location='cpu')
if 'state_dict' in pre2:
    pre2 = pre2['state_dict']


scratch_dict = model.state_dict()
print("<<<<<<<<<<<< scratch dict <<<<<<<<<<<<<")
# print(scratch_dict.keys())

# print(pretrained_weights.keys())

# k1 = []
# for k in scratch_dict.keys():
#     if k not in pretrained_weights.keys():
#         k1.append(k)
# print(k1)

# print("<<<<<<<<<<<< pretrained_weights <<<<<<<<<<<<<")
# k2 = []
# for k in pretrained_weights.keys():
#     if k not in scratch_dict.keys():
#         k2.append(k)
# print(k2)

model.load_state_dict(pretrained_weights, strict=True)
print('>>>>>>>>>>>>>>>>>>>load results>>>>>>>>>>>>>>>>>>>>>>>')
all_zero_params = []
non_zero_params = []
different_params = []
same_params = []
trainable_params = []
un_params = []
for name, param in model.named_parameters():
    if 'control_model' in name:
        print(name, param.size(), torch.sum(param))
        # print(pre2[name], pre2[name].size)
        if abs(torch.sum(param)) < 1e-4:
            all_zero_params.append(name)
        else:
            non_zero_params.append(name)
        # print(param.detach())
        # print(pre2[name].detach())
        if name == 'control_model.input_blocks.0.0.weight':
            p1 = param.detach()
            p2 = pre2[name].detach()
        if abs(torch.sum(param.detach() - pre2[name].detach())) >= 1e-4:
            different_params.append(name)
        else:
            same_params.append(name)
        if param.requires_grad:
            trainable_params.append(name)
        else:
            un_params.append(name)
print('>>>>>>>>>>>>>>>>>>>zero params>>>>>>>>>>>>>>>>>>>>>>>')
print(all_zero_params)
print('>>>>>>>>>>>>>>>>>>>non zero params>>>>>>>>>>>>>>>>>>>>>>>')
print(non_zero_params)
print('>>>>>>>>>>>>>>>>>>>different params>>>>>>>>>>>>>>>>>>>>>>>')
print(different_params)
print('>>>>>>>>>>>>>>>>>>>same params>>>>>>>>>>>>>>>>>>>>>>>')
print(same_params)
print('>>>>>>>>>>>>>>>>>>>trainable params>>>>>>>>>>>>>>>>>>>>>>>')
print(trainable_params)
print('>>>>>>>>>>>>>>>>>>>untrainable params>>>>>>>>>>>>>>>>>>>>>>>')
print(un_params)


# model2.load_state_dict(pre2, strict=True)




# target_dict = {}
# for k in scratch_dict.keys():
#     is_control, name = get_node_name(k, 'control_')
#     if is_control:
#         copy_k = 'model.diffusion_' + name
#     else:
#         copy_k = k
#     if copy_k in pretrained_weights:
#         target_dict[k] = pretrained_weights[copy_k].clone()
#     else:
#         target_dict[k] = scratch_dict[k].clone()
#         print(f'These weights are newly added: {k}')

# model.load_state_dict(target_dict, strict=True)
# torch.save(model.state_dict(), output_path)
print('Done.')
print(len(non_zero_params), len(all_zero_params), len(different_params), len(same_params))
print(p1)
print(p2)
# print(torch.sum(p1), torch.sum(p2), p3)