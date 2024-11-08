import torch
import torch.nn as nn
from torch.nn import functional as func


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out     
        

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        # self.down = nn.MaxPool2d((2, 2), stride=2)
        self.down = nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False, padding=0)
        # self.conv = Conv_Block(in_channels, out_channels)
        
    def forward(self, x):
        d_x = self.down(x)
        # out = self.conv(d_x)
        return d_x
        

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
        self.conv = Conv_Block(in_channels * 2, out_channels)

    def forward(self, x, feature_map):
        u_x = self.up(x)
        # feature_map = crop_img(feature_map, out)
        diff_y = feature_map.size()[2] - u_x.size()[2]
        diff_x = feature_map.size()[3] - u_x.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        u_x = func.pad(u_x, [diff_x // 2, diff_x - diff_x // 2,
                             diff_y // 2, diff_y - diff_y // 2])
        c_x = torch.cat((feature_map, u_x), dim=1)
        out = self.conv(c_x)
        return out
        
class UpSample_d2_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample_d2_1, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

    def forward(self, x):
        u_x = self.up(x)

        return u_x
        
        
class UpSample_d2_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample_d2_2, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
        self.up2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(2, 2), stride=2)

    def forward(self, x):
        u_x = self.up2(self.up1(x))

        return u_x
        

class LMFR_Net_B(nn.Module):
    def __init__(self,
                 base_c = 16,
                 num_classes = 2):
        super(LMFR_Net_B, self).__init__()
        # encoder
        self.e1 = Conv_Block(3, base_c)
        self.down = DownSample(base_c)
        self.e2 = Conv_Block(base_c, base_c)
        #self.d1_2 = DownSample()
        self.e3 = Conv_Block(base_c, base_c)
        
        
        # decoder_1
        self.u1_1 = UpSample(base_c, base_c)
        self.u1_2 = UpSample(base_c, base_c)
        # decoder_2
        self.d2_1 = Conv_Block(base_c, base_c)
        self.u2_1 = UpSample_d2_2(base_c, base_c)
        self.d2_2 = Conv_Block(base_c, base_c)
        self.u2_2 = UpSample_d2_1(base_c, base_c)
        self.d2_3 = Conv_Block(base_c, base_c)
        
        # out
        # self.conv_squeeze = nn.Conv2d(base_c * 3, base_c, kernel_size=(1, 1), padding=0)
        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=(1, 1), padding=0)
        
        self.active = nn.Sigmoid()

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.down(x1))
        x3 = self.e3(self.down(x2))

        x4 = self.u1_1(x3, x2)
        x5 = self.u1_2(x4, x1)

        x6 = self.d2_1(x3)  # 64->32
        x7 = self.d2_2(x4)  # 32->16
        x8 = self.d2_3(x5)
        
        
        #########################################################################
        c2 = self.u2_2(x7)  # 16->16
        c3 = self.u2_1(x6)  # 32->16
        # feature_map = crop_img(feature_map, out)
        diff_y1 = x8.size()[2] - c2.size()[2]
        diff_x1 = x8.size()[3] - c2.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        c2 = func.pad(c2, [diff_x1 // 2, diff_x1 - diff_x1 // 2,
                             diff_y1 // 2, diff_y1 - diff_y1 // 2])
        # feature_map = crop_img(feature_map, out)
        diff_y2 = x8.size()[2] - c3.size()[2]
        diff_x2 = x8.size()[3] - c3.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        c3 = func.pad(c3, [diff_x2 // 2, diff_x2 - diff_x2 // 2,
                             diff_y2 // 2, diff_y2 - diff_y2 // 2])
                             
        #########################################################################
                             
        # x9 = torch.cat([x8, c2, c3], dim=1)
        out1 = self.active(x8)
        out2 = self.active(c2)
        out3 = self.active(c3)
        out = 0.5 * out1 + 0.3 * out2 + 0.2 * out3
        # x10 = self.conv_squeeze(x9)

        out = self.out_conv(out)

        return {"out": out}
        # return out


if __name__ == '__main__':
    x = torch.rand([2, 3, 51, 49])
    net = LMFR_Net_B()
    print(net(x)['out'].shape)


