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
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
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
        # st()
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

        # x1.shape torch.Size([1, 64, 128, 240])
        # x2.shape torch.Size([1, 128, 64, 120])
        # x3.shape torch.Size([1, 256, 32, 60])
        # x4.shape torch.Size([1, 512, 16, 30])
        # x5.shape torch.Size([1, 512, 8, 15])
        # x.shape torch.Size([1, 256, 16, 30])
        # x.shape torch.Size([1, 128, 32, 60])
        # x.shape torch.Size([1, 64, 64, 120])
        # x.shape torch.Size([1, 64, 128, 240])
        # x.shape torch.Size([1, 64, 128, 240])


        return logits

