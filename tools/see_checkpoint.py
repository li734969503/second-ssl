import torch


state_dict = torch.load("/home/jingyu/3d/OpenPCDet/output/ssl/second/default/ckpt/checkpoint_epoch_80.pth")
print(type(state_dict))
 
for i in state_dict:
    print(i)
    print(type(state_dict[i]))
    #print("aa:",state_dict[i].data.size())
    #print("bb:",state_dict[i].requires_grad)
print(state_dict['model_state'])

 