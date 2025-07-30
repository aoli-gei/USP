import torch
import torch.nn as nn

class Universal_Suppressor(nn.Module):
    def Quantization_Awareness_Module(self, input_image):
        input_image_round = torch.round(input_image*255)/255
        return input_image + (input_image_round - input_image.detach())

    def Unilateral_Suppression_Module(self, input_image, n):
        mask_1 = input_image > 1
        input_image[mask_1] = 1 + torch.pow((input_image[mask_1]-1), n)

        mask_0 = input_image < 0
        input_image[mask_0] = -torch.pow(torch.abs(input_image[mask_0]), n)

        return input_image

    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def forward(self, x):
        x = self.Unilateral_Suppression_Module(x, self.n)
        x = self.Quantization_Awareness_Module(x)
        return x