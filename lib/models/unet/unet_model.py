""" Full assembly of the parts to form the complete network """

from .unet_parts import *

from pdb import set_trace as st


class UNetORG(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UNet(nn.Module):
    def __init__(self, unet_type, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.unet_type = unet_type
        if 'i' in self.unet_type:
            n_channels = 4
        else:
            n_channels = 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)

        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down6 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)

        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32 // factor, bilinear)

        self.up6 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        # print(f'count_parameters(self.inc) {count_parameters(self.inc)}')
        # print(f'count_parameters(self.down1) {count_parameters(self.down1)}')
        # print(f'count_parameters(self.down2) {count_parameters(self.down2)}')
        # print(f'count_parameters(self.down3) {count_parameters(self.down3)}')
        # print(f'count_parameters(self.down4) {count_parameters(self.down4)}')
        # print(f'count_parameters(self.down3) {count_parameters(self.down5)}')
        # print(f'count_parameters(self.down4) {count_parameters(self.down6)}')
        # print(f'count_parameters(self.up1) {count_parameters(self.up1)}')
        # print(f'count_parameters(self.up2) {count_parameters(self.up2)}')
        # print(f'count_parameters(self.up3) {count_parameters(self.up3)}')
        # print(f'count_parameters(self.up4) {count_parameters(self.up4)}')
        # print(f'count_parameters(self.up3) {count_parameters(self.up5)}')
        # print(f'count_parameters(self.up4) {count_parameters(self.up6)}')
        # print(f'count_parameters(self.outc) {count_parameters(self.outc)}')

        ### 没有init weight

    def forward(self, heatmaps, aligned_depth_fill, view):
        st()
        x = torch.cat((view, aligned_depth_fill), 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        logits = self.outc(x)

        # (Pdb) x.shape
        # torch.Size([1, 4, 512, 960])
        # (Pdb) x1.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) x2.shape
        # torch.Size([1, 32, 256, 480])
        # (Pdb) x3.shape
        # torch.Size([1, 64, 128, 240])
        # (Pdb) x4.shape
        # torch.Size([1, 128, 64, 120])
        # (Pdb) x5.shape
        # torch.Size([1, 256, 32, 60])
        # (Pdb) x6.shape
        # torch.Size([1, 512, 16, 30])
        # (Pdb) x7.shape
        # torch.Size([1, 512, 8, 15])

        # (Pdb) x.shape
        # torch.Size([1, 256, 16, 30])
        # (Pdb) x.shape
        # torch.Size([1, 128, 32, 60])
        # (Pdb) x.shape
        # torch.Size([1, 64, 64, 120])
        # (Pdb) x.shape
        # torch.Size([1, 32, 128, 240])
        # (Pdb) x.shape
        # torch.Size([1, 16, 256, 480])
        # (Pdb) x.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) logits.shape
        # torch.Size([1, 1, 512, 960])


        return logits

class UNet_LCC(nn.Module):
    def __init__(self, unet_type, n_classes, bilinear=True):
        super(UNet_LCC, self).__init__()
        joint_num = 15
        self.unet_type = unet_type
        if 'i' in self.unet_type:
            n_channels = 4
        else:
            n_channels = 1

        if self.unet_type == 'dbh':
            n_channels = 1 + joint_num
        if self.unet_type == 'dmh':
            n_channels = 1

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        
        if self.unet_type == 'dmh':
            self.down3 = Down(64 + joint_num, 128)
        else:
            self.down3 = Down(64, 128)

        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down6 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)

        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32 // factor, bilinear)

        self.up6 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        print(f'count_parameters(self.inc) {count_parameters(self.inc)}')
        print(f'count_parameters(self.down1) {count_parameters(self.down1)}')
        print(f'count_parameters(self.down2) {count_parameters(self.down2)}')
        print(f'count_parameters(self.down3) {count_parameters(self.down3)}')
        print(f'count_parameters(self.down4) {count_parameters(self.down4)}')
        print(f'count_parameters(self.down3) {count_parameters(self.down5)}')
        print(f'count_parameters(self.down4) {count_parameters(self.down6)}')
        print(f'count_parameters(self.up1) {count_parameters(self.up1)}')
        print(f'count_parameters(self.up2) {count_parameters(self.up2)}')
        print(f'count_parameters(self.up3) {count_parameters(self.up3)}')
        print(f'count_parameters(self.up4) {count_parameters(self.up4)}')
        print(f'count_parameters(self.up3) {count_parameters(self.up5)}')
        print(f'count_parameters(self.up4) {count_parameters(self.up6)}')
        print(f'count_parameters(self.outc) {count_parameters(self.outc)}')
        ### 没有init weight
        

        # count_parameters(self.inc) 2544
        # count_parameters(self.down1) 14016
        # count_parameters(self.down2) 55680
        # count_parameters(self.down3) 221952
        # count_parameters(self.down4) 886272
        # count_parameters(self.down3) 3542016
        # count_parameters(self.down4) 4721664
        # count_parameters(self.up1) 5900544
        # count_parameters(self.up2) 1475712
        # count_parameters(self.up3) 369216
        # count_parameters(self.up4) 92448
        # count_parameters(self.up3) 23184
        # count_parameters(self.up4) 7008
        # count_parameters(self.outc) 17

        # count_parameters(self.inc) 2544
        # count_parameters(self.down1) 14016
        # count_parameters(self.down2) 55680
        # count_parameters(self.down3) 221952
        # count_parameters(self.down4) 886272
        # count_parameters(self.down3) 3542016
        # count_parameters(self.down4) 4721664
        # count_parameters(self.up1) 5900544
        # count_parameters(self.up2) 1475712
        # count_parameters(self.up3) 369216
        # count_parameters(self.up4) 92448
        # count_parameters(self.up3) 23184
        # count_parameters(self.up4) 7008
        # count_parameters(self.outc) 255   

    def forward(self, heatmaps, aligned_depth_fill, view):
        # st()
        if self.unet_type in ['d', 'dmh']:
            x = aligned_depth_fill
        elif self.unet_type == 'di':
            x = torch.cat((view, aligned_depth_fill), 1)
        elif self.unet_type == 'dbh':
            heatmaps_up = F.upsample(heatmaps, scale_factor=4, mode='bicubic', align_corners=True)
            x = torch.cat((heatmaps_up, aligned_depth_fill), 1)


        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        if self.unet_type == 'dmh':
            x3_cat = torch.cat((heatmaps, x3), 1)
            x4 = self.down3(x3_cat)
        else:
            x4 = self.down3(x3)

        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        
        logits = self.outc(x)

        # st() 检查dep15爆炸原因，这里不是显存爆炸的原因，毕竟只多了一点参数
        # logits = logits[:,0:1,:,:]

        # (Pdb) x.shape
        # torch.Size([1, 4, 512, 960])
        # (Pdb) x1.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) x2.shape
        # torch.Size([1, 32, 256, 480])
        # (Pdb) x3.shape
        # torch.Size([1, 64, 128, 240])
        # (Pdb) x4.shape
        # torch.Size([1, 128, 64, 120])
        # (Pdb) x5.shape
        # torch.Size([1, 256, 32, 60])
        # (Pdb) x6.shape
        # torch.Size([1, 512, 16, 30])
        # (Pdb) x7.shape
        # torch.Size([1, 512, 8, 15])

        # (Pdb) x.shape
        # torch.Size([1, 256, 16, 30])
        # (Pdb) x.shape
        # torch.Size([1, 128, 32, 60])
        # (Pdb) x.shape
        # torch.Size([1, 64, 64, 120])
        # (Pdb) x.shape
        # torch.Size([1, 32, 128, 240])
        # (Pdb) x.shape
        # torch.Size([1, 16, 256, 480])
        # (Pdb) x.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) logits.shape
        # torch.Size([1, 1, 512, 960])
        return logits



class UNet_LCC_MSH(nn.Module):
    def __init__(self, unet_type, n_classes, bilinear=True):
        super(UNet_LCC_MSH, self).__init__()
        joint_num = 15
        self.unet_type = unet_type
        if 'i' in self.unet_type:
            n_channels = 4
        else:
            n_channels = 1

        if self.unet_type in ['dbh', 'dmsh']:
            n_channels = 1 + joint_num
        if self.unet_type == 'dmh':
            n_channels = 1
        
        if self.unet_type == 'dmsh':
            add_heatmap_channel = joint_num
        else:
            add_heatmap_channel = 0


        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16 + add_heatmap_channel, 32)
        self.down2 = Down(32 + add_heatmap_channel, 64)
        
        if self.unet_type == 'dmh':
            self.down3 = Down(64 + joint_num, 128)
        else:
            self.down3 = Down(64 + add_heatmap_channel, 128)

        self.down4 = Down(128 + add_heatmap_channel, 256)
        self.down5 = Down(256 + add_heatmap_channel, 512)
        factor = 2 if bilinear else 1
        self.down6 = Down(512 + add_heatmap_channel, 1024 // factor)

   
        up_func = Up_LCC
        # st()

        self.up1 = up_func(1024 + add_heatmap_channel, 512 // factor, bilinear)
        self.up2 = up_func(512 + add_heatmap_channel, 256 // factor, bilinear)
        self.up3 = up_func(256 + add_heatmap_channel, 128 // factor, bilinear)

        self.up4 = up_func(128 + add_heatmap_channel, 64 // factor, bilinear)
        self.up5 = up_func(64 + add_heatmap_channel, 32 // factor, bilinear)

        self.up6 = up_func(32 + add_heatmap_channel, 16, bilinear)

        self.outc = OutConv(16 + add_heatmap_channel, n_classes)

        # print(f'count_parameters(self.inc) {count_parameters(self.inc)}')
        # print(f'count_parameters(self.down1) {count_parameters(self.down1)}')
        # print(f'count_parameters(self.down2) {count_parameters(self.down2)}')
        # print(f'count_parameters(self.down3) {count_parameters(self.down3)}')
        # print(f'count_parameters(self.down4) {count_parameters(self.down4)}')
        # print(f'count_parameters(self.down3) {count_parameters(self.down5)}')
        # print(f'count_parameters(self.down4) {count_parameters(self.down6)}')
        # print(f'count_parameters(self.up1) {count_parameters(self.up1)}')
        # print(f'count_parameters(self.up2) {count_parameters(self.up2)}')
        # print(f'count_parameters(self.up3) {count_parameters(self.up3)}')
        # print(f'count_parameters(self.up4) {count_parameters(self.up4)}')
        # print(f'count_parameters(self.up3) {count_parameters(self.up5)}')
        # print(f'count_parameters(self.up4) {count_parameters(self.up6)}')
        # print(f'count_parameters(self.outc) {count_parameters(self.outc)}')

        ### 没有init weight
        
        

    def forward(self, heatmaps, aligned_depth_fill, view):
        # st()
        if self.unet_type in ['d', 'dmh']:
            x = aligned_depth_fill
        elif self.unet_type == 'di':
            x = torch.cat((view, aligned_depth_fill), 1)
        elif self.unet_type == 'dbh':
            heatmaps_up = F.upsample(heatmaps, scale_factor=4, mode='bicubic', align_corners=True)
            x = torch.cat((heatmaps_up, aligned_depth_fill), 1)
        elif self.unet_type == 'dmsh':
            heatmaps_512x960 = F.upsample(heatmaps, scale_factor=4, mode='bicubic', align_corners=True)
            heatmaps_256x480 = F.upsample(heatmaps, scale_factor=2, mode='bicubic', align_corners=True)
            heatmaps_128x240 = heatmaps
            heatmaps_64x120 = F.interpolate(heatmaps, size=[64, 120], mode='bicubic', align_corners=True)
            heatmaps_32x60 = F.interpolate(heatmaps, size=[32, 60], mode='bicubic', align_corners=True)
            heatmaps_16x30 = F.interpolate(heatmaps, size=[16, 30], mode='bicubic', align_corners=True)
            heatmaps_8x15 = F.interpolate(heatmaps, size=[8, 15], mode='bicubic', align_corners=True)
            x = torch.cat((heatmaps_512x960, aligned_depth_fill), 1)
            
        x1 = self.inc(x)

        if self.unet_type == 'dmsh':
            x1_cat = torch.cat((heatmaps_512x960, x1), dim=1)
            x2 = self.down1(x1_cat)
        else:
            x2 = self.down1(x1)

        if self.unet_type == 'dmsh':
            x2_cat = torch.cat((heatmaps_256x480, x2), dim=1)
            x3 = self.down2(x2_cat)
        else:
            x3 = self.down2(x2)

        if self.unet_type in ['dmh', 'dmsh']:
            x3_cat = torch.cat((heatmaps, x3), 1)
            x4 = self.down3(x3_cat)
        else:
            x4 = self.down3(x3)

        if self.unet_type == 'dmsh':
            x4_cat = torch.cat((heatmaps_64x120, x4), dim=1)
            x5 = self.down4(x4_cat)
        else:
            x5 = self.down4(x4)

        if self.unet_type == 'dmsh':
            x5_cat = torch.cat((heatmaps_32x60, x5), dim=1)
            x6 = self.down5(x5_cat)
        else:
            x6 = self.down5(x5)

        if self.unet_type == 'dmsh':
            x6_cat = torch.cat((heatmaps_16x30, x6), dim=1)
            x7 = self.down6(x6_cat)
        else:
            x7 = self.down6(x6)

        if self.unet_type == 'dmsh':
            x = self.up1(x7, x6, heatmaps_16x30)
            x = self.up2(x, x5, heatmaps_32x60)
            x = self.up3(x, x4, heatmaps_64x120)
            x = self.up4(x, x3, heatmaps)
            x = self.up5(x, x2, heatmaps_256x480)
            x = self.up6(x, x1, heatmaps_512x960)
        else:
            x = self.up1(x7, x6)
            x = self.up2(x, x5)
            x = self.up3(x, x4)
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)

        if self.unet_type == 'dmsh':
            x_cat = torch.cat((heatmaps_512x960, x), dim=1)
            logits = self.outc(x_cat)
        else:
            logits = self.outc(x)

        # (Pdb) x.shape
        # torch.Size([1, 4, 512, 960])
        # (Pdb) x1.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) x2.shape
        # torch.Size([1, 32, 256, 480])
        # (Pdb) x3.shape
        # torch.Size([1, 64, 128, 240])
        # (Pdb) x4.shape
        # torch.Size([1, 128, 64, 120])
        # (Pdb) x5.shape
        # torch.Size([1, 256, 32, 60])
        # (Pdb) x6.shape
        # torch.Size([1, 512, 16, 30])
        # (Pdb) x7.shape
        # torch.Size([1, 512, 8, 15])

        # (Pdb) x.shape
        # torch.Size([1, 256, 16, 30])
        # (Pdb) x.shape
        # torch.Size([1, 128, 32, 60])
        # (Pdb) x.shape
        # torch.Size([1, 64, 64, 120])
        # (Pdb) x.shape
        # torch.Size([1, 32, 128, 240])
        # (Pdb) x.shape
        # torch.Size([1, 16, 256, 480])
        # (Pdb) x.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) logits.shape
        # torch.Size([1, 1, 512, 960])


        return logits



class UNet_LCC_1920x1024(nn.Module):
    def __init__(self, unet_type, n_classes, bilinear=True):
        super(UNet_LCC_1920x1024, self).__init__()
        joint_num = 15
        self.unet_type = unet_type
        if 'i' in self.unet_type:
            n_channels = 4
        else:
            n_channels = 1

        if self.unet_type == 'dbh':
            n_channels = 1 + joint_num
        if self.unet_type == 'dmh':
            n_channels = 1

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)

        self.down0 = Down(8, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        
        if self.unet_type == 'dmh':
            self.down3 = Down(64 + joint_num, 128)
        else:
            self.down3 = Down(64, 128)

        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down6 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32 // factor, bilinear)
        self.up6 = Up(32, 16 // factor, bilinear)
        self.up7 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

        print(f'count_parameters(self.inc) {count_parameters(self.inc)}')
        print(f'count_parameters(self.down1) {count_parameters(self.down1)}')
        print(f'count_parameters(self.down2) {count_parameters(self.down2)}')
        print(f'count_parameters(self.down3) {count_parameters(self.down3)}')
        print(f'count_parameters(self.down4) {count_parameters(self.down4)}')
        print(f'count_parameters(self.down3) {count_parameters(self.down5)}')
        print(f'count_parameters(self.down4) {count_parameters(self.down6)}')
        print(f'count_parameters(self.up1) {count_parameters(self.up1)}')
        print(f'count_parameters(self.up2) {count_parameters(self.up2)}')
        print(f'count_parameters(self.up3) {count_parameters(self.up3)}')
        print(f'count_parameters(self.up4) {count_parameters(self.up4)}')
        print(f'count_parameters(self.up3) {count_parameters(self.up5)}')
        print(f'count_parameters(self.up4) {count_parameters(self.up6)}')
        print(f'count_parameters(self.outc) {count_parameters(self.outc)}')
        ### 没有init weight
        

        # count_parameters(self.inc) 2544
        # count_parameters(self.down1) 14016
        # count_parameters(self.down2) 55680
        # count_parameters(self.down3) 221952
        # count_parameters(self.down4) 886272
        # count_parameters(self.down3) 3542016
        # count_parameters(self.down4) 4721664
        # count_parameters(self.up1) 5900544
        # count_parameters(self.up2) 1475712
        # count_parameters(self.up3) 369216
        # count_parameters(self.up4) 92448
        # count_parameters(self.up3) 23184
        # count_parameters(self.up4) 7008
        # count_parameters(self.outc) 17

        # count_parameters(self.inc) 2544
        # count_parameters(self.down1) 14016
        # count_parameters(self.down2) 55680
        # count_parameters(self.down3) 221952
        # count_parameters(self.down4) 886272
        # count_parameters(self.down3) 3542016
        # count_parameters(self.down4) 4721664
        # count_parameters(self.up1) 5900544
        # count_parameters(self.up2) 1475712
        # count_parameters(self.up3) 369216
        # count_parameters(self.up4) 92448
        # count_parameters(self.up3) 23184
        # count_parameters(self.up4) 7008
        # count_parameters(self.outc) 255   

    def forward(self, heatmaps, aligned_depth_fill, view):
        # st()
        if self.unet_type in ['d', 'dmh']:
            x = aligned_depth_fill
        elif self.unet_type == 'di':
            x = torch.cat((view, aligned_depth_fill), 1)
        elif self.unet_type == 'dbh':
            heatmaps_up = F.upsample(heatmaps, scale_factor=4, mode='bicubic', align_corners=True)
            x = torch.cat((heatmaps_up, aligned_depth_fill), 1)


        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        if self.unet_type == 'dmh':
            x3_cat = torch.cat((heatmaps, x3), 1)
            x4 = self.down3(x3_cat)
        else:
            x4 = self.down3(x3)

        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.up7(x, x0)
        
        logits = self.outc(x)

        # st() 检查dep15爆炸原因，这里不是显存爆炸的原因，毕竟只多了一点参数
        # logits = logits[:,0:1,:,:]

        # (Pdb) x.shape
        # torch.Size([1, 4, 512, 960])
        # (Pdb) x1.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) x2.shape
        # torch.Size([1, 32, 256, 480])
        # (Pdb) x3.shape
        # torch.Size([1, 64, 128, 240])
        # (Pdb) x4.shape
        # torch.Size([1, 128, 64, 120])
        # (Pdb) x5.shape
        # torch.Size([1, 256, 32, 60])
        # (Pdb) x6.shape
        # torch.Size([1, 512, 16, 30])
        # (Pdb) x7.shape
        # torch.Size([1, 512, 8, 15])

        # (Pdb) x.shape
        # torch.Size([1, 256, 16, 30])
        # (Pdb) x.shape
        # torch.Size([1, 128, 32, 60])
        # (Pdb) x.shape
        # torch.Size([1, 64, 64, 120])
        # (Pdb) x.shape
        # torch.Size([1, 32, 128, 240])
        # (Pdb) x.shape
        # torch.Size([1, 16, 256, 480])
        # (Pdb) x.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) logits.shape
        # torch.Size([1, 1, 512, 960])
        return logits


class UNet_LCC_MSH_1920x1024(nn.Module):
    def __init__(self, unet_type, n_classes, bilinear=True):
        super(UNet_LCC_MSH_1920x1024, self).__init__()
        joint_num = 15
        self.unet_type = unet_type
        if 'i' in self.unet_type:
            n_channels = 4
        else:
            n_channels = 1

        if self.unet_type in ['dbh', 'dmsh']:
            n_channels = 1 + joint_num
        if self.unet_type == 'dmh':
            n_channels = 1
        
        if self.unet_type == 'dmsh':
            add_heatmap_channel = joint_num
        else:
            add_heatmap_channel = 0


        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 16)
        self.inc = DoubleConv(n_channels, 8)

        self.down0 = Down(8 + add_heatmap_channel, 16)

        self.down1 = Down(16 + add_heatmap_channel, 32)
        self.down2 = Down(32 + add_heatmap_channel, 64)
        
        if self.unet_type == 'dmh':
            self.down3 = Down(64 + joint_num, 128)
        else:
            self.down3 = Down(64 + add_heatmap_channel, 128)

        self.down4 = Down(128 + add_heatmap_channel, 256)
        
        factor = 2 if bilinear else 1
        self.down5 = Down(256 + add_heatmap_channel, 512 // factor)
        

        # self.down6 = Down(512 + add_heatmap_channel, 1024 // factor)

   
        up_func = Up_LCC
        # st()

        # self.up1 = up_func(1024 + add_heatmap_channel, 512 // factor, bilinear)
        self.up2 = up_func(512 + add_heatmap_channel, 256 // factor, bilinear)
        self.up3 = up_func(256 + add_heatmap_channel, 128 // factor, bilinear)

        self.up4 = up_func(128 + add_heatmap_channel, 64 // factor, bilinear)
        self.up5 = up_func(64 + add_heatmap_channel, 32 // factor, bilinear)

        self.up6 = up_func(32 + add_heatmap_channel, 16 // factor, bilinear)
        self.up7 = up_func(16 + add_heatmap_channel, 8, bilinear)

        self.outc = OutConv(8 + add_heatmap_channel, n_classes)

        # print(f'count_parameters(self.inc) {count_parameters(self.inc)}')
        # print(f'count_parameters(self.down1) {count_parameters(self.down1)}')
        # print(f'count_parameters(self.down2) {count_parameters(self.down2)}')
        # print(f'count_parameters(self.down3) {count_parameters(self.down3)}')
        # print(f'count_parameters(self.down4) {count_parameters(self.down4)}')
        # print(f'count_parameters(self.down3) {count_parameters(self.down5)}')
        # print(f'count_parameters(self.down4) {count_parameters(self.down6)}')
        # print(f'count_parameters(self.up1) {count_parameters(self.up1)}')
        # print(f'count_parameters(self.up2) {count_parameters(self.up2)}')
        # print(f'count_parameters(self.up3) {count_parameters(self.up3)}')
        # print(f'count_parameters(self.up4) {count_parameters(self.up4)}')
        # print(f'count_parameters(self.up3) {count_parameters(self.up5)}')
        # print(f'count_parameters(self.up4) {count_parameters(self.up6)}')
        # print(f'count_parameters(self.outc) {count_parameters(self.outc)}')

        ### 没有init weight
        
        

    def forward(self, heatmaps, aligned_depth_fill, view):
        # st()
        if self.unet_type in ['d', 'dmh']:
            x = aligned_depth_fill
        elif self.unet_type == 'di':
            x = torch.cat((view, aligned_depth_fill), 1)
        elif self.unet_type == 'dbh':
            heatmaps_up = F.upsample(heatmaps, scale_factor=4, mode='bicubic', align_corners=True)
            x = torch.cat((heatmaps_up, aligned_depth_fill), 1)
        elif self.unet_type == 'dmsh':
            heatmaps_1024x1920 = F.upsample(heatmaps, scale_factor=8, mode='bicubic', align_corners=True)
            heatmaps_512x960 = F.upsample(heatmaps, scale_factor=4, mode='bicubic', align_corners=True)
            heatmaps_256x480 = F.upsample(heatmaps, scale_factor=2, mode='bicubic', align_corners=True)
            heatmaps_128x240 = heatmaps
            heatmaps_64x120 = F.interpolate(heatmaps, size=[64, 120], mode='bicubic', align_corners=True)
            heatmaps_32x60 = F.interpolate(heatmaps, size=[32, 60], mode='bicubic', align_corners=True)
            heatmaps_16x30 = F.interpolate(heatmaps, size=[16, 30], mode='bicubic', align_corners=True)
            heatmaps_8x15 = F.interpolate(heatmaps, size=[8, 15], mode='bicubic', align_corners=True)
            x = torch.cat((heatmaps_1024x1920, aligned_depth_fill), 1)
            
        # x1 = self.inc(x)
        x0 = self.inc(x)
        if self.unet_type == 'dmsh':
            x0_cat = torch.cat((heatmaps_1024x1920, x0), dim=1)
            x1 = self.down0(x0_cat)
        else:
            x1 = self.down0(x0)

        if self.unet_type == 'dmsh':
            x1_cat = torch.cat((heatmaps_512x960, x1), dim=1)
            x2 = self.down1(x1_cat)
        else:
            x2 = self.down1(x1)

        if self.unet_type == 'dmsh':
            x2_cat = torch.cat((heatmaps_256x480, x2), dim=1)
            x3 = self.down2(x2_cat)
        else:
            x3 = self.down2(x2)

        if self.unet_type in ['dmh', 'dmsh']:
            x3_cat = torch.cat((heatmaps, x3), 1)
            x4 = self.down3(x3_cat)
        else:
            x4 = self.down3(x3)

        if self.unet_type == 'dmsh':
            x4_cat = torch.cat((heatmaps_64x120, x4), dim=1)
            x5 = self.down4(x4_cat)
        else:
            x5 = self.down4(x4)

        if self.unet_type == 'dmsh':
            x5_cat = torch.cat((heatmaps_32x60, x5), dim=1)
            x6 = self.down5(x5_cat)
        else:
            x6 = self.down5(x5)

        # if self.unet_type == 'dmsh':
        #     x6_cat = torch.cat((heatmaps_16x30, x6), dim=1)
        #     x7 = self.down6(x6_cat)
        # else:
        #     x7 = self.down6(x6)

        if self.unet_type == 'dmsh':
            # # ### w down6
            # x = self.up1(x7, x6, heatmaps_16x30)
            # x = self.up2(x, x5, heatmaps_32x60)

            x = self.up2(x6, x5, heatmaps_32x60)


            x = self.up3(x, x4, heatmaps_64x120)
            x = self.up4(x, x3, heatmaps)
            x = self.up5(x, x2, heatmaps_256x480)
            x = self.up6(x, x1, heatmaps_512x960)
            x = self.up7(x, x0, heatmaps_1024x1920)
        else:
            # ### w down6
            # # x = self.up1(x7, x6)
            # x = self.up2(x, x5)

            x = self.up2(x6, x5)

            x = self.up3(x, x4)
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)
            x = self.up7(x, x0)

        # if self.unet_type == 'dmsh':
        #     x_cat = torch.cat((heatmaps_512x960, x), dim=1)
        #     logits = self.outc(x_cat)
        # else:
        #     logits = self.outc(x)

        if self.unet_type == 'dmsh':
            x_cat = torch.cat((heatmaps_1024x1920, x), dim=1)
            logits = self.outc(x_cat)
        else:
            logits = self.outc(x)

        # (Pdb) x.shape
        # torch.Size([1, 4, 512, 960])
        # (Pdb) x1.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) x2.shape
        # torch.Size([1, 32, 256, 480])
        # (Pdb) x3.shape
        # torch.Size([1, 64, 128, 240])
        # (Pdb) x4.shape
        # torch.Size([1, 128, 64, 120])
        # (Pdb) x5.shape
        # torch.Size([1, 256, 32, 60])
        # (Pdb) x6.shape
        # torch.Size([1, 512, 16, 30])
        # (Pdb) x7.shape
        # torch.Size([1, 512, 8, 15])

        # (Pdb) x.shape
        # torch.Size([1, 256, 16, 30])
        # (Pdb) x.shape
        # torch.Size([1, 128, 32, 60])
        # (Pdb) x.shape
        # torch.Size([1, 64, 64, 120])
        # (Pdb) x.shape
        # torch.Size([1, 32, 128, 240])
        # (Pdb) x.shape
        # torch.Size([1, 16, 256, 480])
        # (Pdb) x.shape
        # torch.Size([1, 16, 512, 960])
        # (Pdb) logits.shape
        # torch.Size([1, 1, 512, 960])


        return logits
