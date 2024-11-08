"""
    论文复现
    双解码器U-Net——DDU-Net
    2022、05、30
"""
import torch
from torch import nn
from torch.nn import functional as func


class Double_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)
        

class Conv_Block_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block_1, self).__init__()
        self.layer= nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x, feature_map):
        up = func.interpolate(x, scale_factor=2, mode='nearest')

        diff_y = feature_map.size()[2] - up.size()[2]
        diff_x = feature_map.size()[3] - up.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        up = func.pad(up, [diff_x // 2, diff_x - diff_x // 2,
                           diff_y // 2, diff_y - diff_y // 2])

        return torch.cat((up, feature_map), dim=1)


class UpSample_De_1(nn.Module):
    def __init__(self, channels):
        super(UpSample_De_1, self).__init__()
        self.layer = nn.Conv2d(channels, channels // 2, kernel_size=1)

    def forward(self, x, feature_map):
        up = func.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)

        diff_y = feature_map.size()[2] - out.size()[2]
        diff_x = feature_map.size()[3] - out.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        out = func.pad(out, [diff_x // 2, diff_x - diff_x // 2,
                             diff_y // 2, diff_y - diff_y // 2])

        return torch.cat((out, feature_map), dim=1)


class UpSample_De_2(nn.Module):
    def __init__(self):
        super(UpSample_De_2, self).__init__()

    def forward(self, x, feature_map_1, feature_map_2):
        up = func.interpolate(x, scale_factor=2, mode='nearest')

        diff_y1 = feature_map_1.size()[2] - up.size()[2]
        diff_x1 = feature_map_1.size()[3] - up.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        up = func.pad(up, [diff_x1 // 2, diff_x1 - diff_x1 // 2,
                           diff_y1 // 2, diff_y1 - diff_y1 // 2])

        diff_y2 = feature_map_1.size()[2] - feature_map_2.size()[2]
        diff_x2 = feature_map_1.size()[3] - feature_map_2.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        feature_map_2 = func.pad(feature_map_2, [diff_x2 // 2, diff_x2 - diff_x2 // 2,
                                                 diff_y2 // 2, diff_y2 - diff_y2 // 2])

        return torch.cat((up, feature_map_1, feature_map_2), dim=1)


# 仅用于解码器-2第一个融合
class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x, feature_map):

        diff_y = x.size()[2] - feature_map.size()[2]
        diff_x = x.size()[3] - feature_map.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        feature_map = func.pad(feature_map, [diff_x // 2, diff_x - diff_x // 2,
                                             diff_y // 2, diff_y - diff_y // 2])

        return torch.cat((x, feature_map), dim=1)


class DDU_Net(nn.Module):
    def __init__(self):
        super(DDU_Net, self).__init__()
        # 根据网络结构进行搭建
        # encode
        self.c1 = Double_Conv_Block(3, 32)      
        self.d1 = DownSample()                  
        self.c2 = Double_Conv_Block(32, 64)    
        # DownSample                            
        self.c3 = Double_Conv_Block(64, 128)  
        # DownSample                            
        self.c4 = Double_Conv_Block(128, 256)   
        # DownSample                            
        self.c5 = Double_Conv_Block(256, 256)   

        # decode_1
        # u1 + c4 = cat1
        self.u1 = UpSample()                    
        self.c6 = Double_Conv_Block(512, 256)   
        # u2 + c3 = cat2
        self.u2 = UpSample_De_1(256)             
        self.c7 = Double_Conv_Block(256, 128)    
        # u3 + c2 = cat3
        self.u3 = UpSample_De_1(128)            
        self.c8 = Double_Conv_Block(128, 64)    
        # u4 + c1 = cat4
        self.u4 = UpSample_De_1(64)             
        self.c9 = Double_Conv_Block(64, 32)     

        # decode_2
        self.cat = Cat()
        self.c10 = Double_Conv_Block(256, 128)    
        self.u5 = UpSample_De_2()               
        self.c11 = Double_Conv_Block(256, 32)    
        # UpSample_De_2                          
        self.c12 = Double_Conv_Block(96, 32)    

        # out
        self.c13 = Conv_Block_1(32, 2)

    def forward(self, x):
        # encoder
        E1 = self.c1(x)
        E2 = self.c2(self.d1(E1))
        E3 = self.c3(self.d1(E2))
        E4 = self.c4(self.d1(E3))
        E5 = self.c5(self.d1(E4))
        #E5 = self.c5(self.dac(self.d1(E4)))
        # decoder_1
        D1_1 = self.c6(self.u1(E5, E4))
        D1_2 = self.c7(self.u2(D1_1, E3))
        D1_3 = self.c8(self.u3(D1_2, E2))
        D1_4 = self.c9(self.u4(D1_3, E1))
        # decoder_2
        D2_1 = self.c10(self.cat(E3, D1_2))
        D2_2 = self.c11(self.u5(D2_1, E2, D1_3))
        D2_3 = self.c12(self.u5(D2_2, E1, D1_4))
        # out
        out = self.c13(D2_3)

        return {"out": out}
        # return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512)
    net = DDU_Attention_Net()
    print(net(x).shape)





