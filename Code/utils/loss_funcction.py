import torch
from torch import nn
import torch.nn.functional as F
from torch import sin, cos
from math import pi


class SetCriterion(nn.Module):
    """ This class computes the loss, which consists of translation and rotation for now."""
    
    def __init__(self,device):
           super().__init__()
           self.device = device
        
    def loss_translation(self, outputs, targets):
        """ calculate the loss related to the translation of pose estimation with L2 Loss"""

        loss_translation = F.mse_loss(outputs, targets, reduction='none')
        loss_translation = torch.sum(loss_translation, dim=1)
        loss_translation = torch.sqrt(loss_translation)
        loss = loss_translation.sum() / len(targets)
        return loss.to(self.device)
    
    def loss_rotation(self, outputs, targets):
        """L = arccos( 0.5 * (Trace(R\tilde(R)^T) -1)
        Calculates the loss in radiant.
        """
        eps = 1e-6

        batch = outputs.shape[0]
        nobj = outputs.shape[1]

        loss_rotation = 0

        for i in range(batch):
            for j in range(nobj):
                src_rot = self.euler2rotm(outputs[i,j,:,0], outputs[i,j,:,1], outputs[i,j,:,2])
                tgt_rot = self.euler2rotm(targets[i,j,:,0], targets[i,j,:,1], targets[i,j,:,2])
                product = torch.matmul(src_rot, tgt_rot.transpose(1, 0))
                trace = torch.sum(product[torch.eye(3).bool()])
                theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
                rad = torch.acos(theta)
                loss_rotation += rad.sum() / len(targets)   
        return loss_rotation
    

    def euler2rotm(self,roll, pitch, yaw):
        """Convert euler angles to rotation matrix."""

        tensor_0 = torch.zeros(1).to(self.device)
        tensor_1 = torch.ones(1).to(self.device)

        RX = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, cos(roll), -sin(roll)]),
                torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3,3)

        RY = torch.stack([
                        torch.stack([cos(pitch), tensor_0, sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3,3)

        RZ = torch.stack([
                        torch.stack([cos(yaw), -sin(yaw), tensor_0]),
                        torch.stack([sin(yaw), cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)

        return R.to(self.device)

    def forward(self, outputs, targets):
        """ This performs the loss computation."""
        loss = self.loss_translation(outputs[:,:,0,:] , targets[:,:,0,:] )
        loss = self.loss_rotation(outputs[:,:,1:,:] , targets[:,:,1:,:] )
        return loss


# T = torch.rand((1,20,2,3))
# K = torch.rand((1,20,2,3))


# loss = SetCriterion(device=torch.device("cuda:0"))
# print(loss(T, K))
