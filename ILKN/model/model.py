from torch import nn
from model.blocks import pixelshuffle_block
from model.blocks import conv_layer
import torch
from model.blocks import InceptionDWConv2d


class ILKN(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.c1 = conv_layer(3, 64, 3)
        self.p1 = InceptionDWConv2d(64)
        self.p2 = InceptionDWConv2d(64)
        self.p3 = InceptionDWConv2d(64)
        self.p4 = InceptionDWConv2d(64)
        self.p5 = InceptionDWConv2d(64)
        self.p6 = InceptionDWConv2d(64)
        self.p7 = InceptionDWConv2d(64)
        self.p8 = InceptionDWConv2d(64)
        self.p9 = InceptionDWConv2d(64)
        self.p10 = InceptionDWConv2d(64)
        self.p11 = InceptionDWConv2d(64)
        self.p12 = InceptionDWConv2d(64)
        self.p13 = InceptionDWConv2d(64)
        self.p14 = InceptionDWConv2d(64)
        self.p15 = InceptionDWConv2d(64)
        self.p16 = InceptionDWConv2d(64)
        self.p17 = InceptionDWConv2d(64)
        self.p18 = InceptionDWConv2d(64)
        self.c2 = conv_layer(64, 64, 1)
        self.upsampler = pixelshuffle_block(64, 3, upscale_factor)

    def forward(self, x):
        out_fea = self.c1(x)
        out_p1 = self.p1(out_fea)
        out_p2 = self.p2(out_p1)
        out_p3 = self.p3(out_p2)
        out_p4 = self.p4(out_p3)
        out_p5 = self.p5(out_p4)
        out_p6 = self.p6(out_p5)
        out_p7 = self.p7(out_p6)
        out_p8 = self.p8(out_p7)
        out_p9 = self.p9(out_p8)
        out_p10 = self.p10(out_p9)
        out_p11 = self.p11(out_p10)
        out_p12 = self.p12(out_p11)
        out_p13 = self.p13(out_p12)
        out_p14 = self.p14(out_p13)
        out_p15 = self.p15(out_p14)
        out_p16 = self.p16(out_p15)
        out_p17 = self.p17(out_p16)
        out_p18 = self.p18(out_p17)
        body_out = self.c2(out_p18)
        out_lr = body_out + out_fea
        out = self.upsampler(out_lr)
        return out
