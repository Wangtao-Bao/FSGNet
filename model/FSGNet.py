import time
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x_source

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * res

def autopad(k, p=None, d=1):

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))

class PConv(nn.Module):

    def __init__(self, c1, c2, k, s):
        super().__init__()

        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class MAIM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MAIM, self).__init__()
        self.conv1 = PConv(in_channels, out_channels, k=3, s=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = PConv(out_channels, out_channels,k=3, s=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out = self.sa(out)
        out += residual
        out = self.relu(out)
        return out



class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.shape
        fft_x = torch.fft.rfft2(x, norm='ortho')
        fft_x = torch.cat([fft_x.real, fft_x.imag], dim=1)
        fft_x = self.conv_layer(fft_x)
        fft_x = self.relu(self.bn(fft_x))
        fft_x = torch.complex(fft_x[:, :c], fft_x[:, c:])
        output = torch.fft.irfft2(fft_x, s=(h, w), norm='ortho')
        return output



class MFM(nn.Module):
    def __init__(self, dim):
        super(MFM, self).__init__()
        self.dim = dim

        self.mixer_gloal = FourierUnit(dim * 2,dim * 2)

        self.conv_init = nn.Conv2d(dim, dim * 2, kernel_size=1)

        self.dw_conv = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, padding_mode='reflect')
            for k in [3, 5]
        ])

        self.ca = ChannelAttention(dim)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect')
        )

    def forward(self, x):
        x = self.conv_init(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x_local = torch.cat([conv(x1) for conv in self.dw_conv], dim=1)
        x_global = self.mixer_gloal(x_local)
        x = self.ca_conv(x_global) + x2
        x = self.ca(x)
        return x

class GPM(nn.Module):
    def __init__(self, k, k_out):
        super(GPM, self).__init__()

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.conv2 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(k, k, 3, 1, 1, bias=False)

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        self.ca = ChannelAttention(k)

    def forward(self, x):
        x_size = x.size()

        y2 = self.conv2(self.pool2(x))
        y2 = self.ca(y2)

        y4 = self.conv4(self.pool4(x))
        y4 = y4 + self.pool(y2)
        y4 = self.ca(y4)

        y8 = self.conv8(self.pool8(x))
        y8 = y8 + self.pool(y4)
        y8 = self.ca(y8)
        resl = torch.add(x, F.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True))
        resl = torch.add(resl, F.interpolate(y4, x_size[2:], mode='bilinear', align_corners=True))
        resl = torch.add(resl, F.interpolate(y8, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        resl = self.conv_sum(resl)
        return resl



class FSGNet(nn.Module):
    def __init__(self,Train=False):
        super().__init__()
        self.Train=Train
        block = MAIM
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [1, 1, 1, 1]
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.MFM1 = MFM(param_channels[0])
        self.MFM2 = MFM(param_channels[1])
        self.MFM3 = MFM(param_channels[2])
        self.MFM4 = MFM(param_channels[3])
        self.encoder_1 = self._make_layer(1, param_channels[0], block)
        self.encoder_2 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_3 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_4 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
        self.encoder_5 = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        self.deeppoollayer = GPM(param_channels[4], param_channels[4])

        self.deeppoolconv4 = nn.Conv2d(param_channels[4], param_channels[3], 1, 1)
        self.deeppoolconv3 = nn.Conv2d(param_channels[4], param_channels[2],1, 1)
        self.deeppoolconv2 = nn.Conv2d(param_channels[4], param_channels[1], 1, 1)
        self.deeppoolconv1 = nn.Conv2d(param_channels[4], param_channels[0], 1, 1)

        self.upconv = nn.Conv2d(param_channels[4], param_channels[3], 1, 1)

        self.decoder_4 = self._make_layer(param_channels[3], param_channels[2], block, param_blocks[3])
        self.decoder_3 = self._make_layer(param_channels[2], param_channels[1], block, param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[1], param_channels[0], block, param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[0], param_channels[0], block)
        self.output_1 = nn.Conv2d(param_channels[0], 1, 1)

        self.output_deeppool = nn.Conv2d(param_channels[4], 1, 1)


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        x_e1 = self.encoder_1(x)
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_e4 = self.encoder_4(self.pool(x_e3))
        x_e5 = self.encoder_5(self.pool(x_e4))

        x_deeppool = self.deeppoollayer(x_e5)

        x_d4 = self.decoder_4(self.MFM4(x_e4) + self.up(self.upconv(x_e5)) + self.up(self.deeppoolconv4(x_deeppool)))
        x_d3 = self.decoder_3(self.MFM3(x_e3) + self.up(x_d4) + self.up_4(self.deeppoolconv3(x_deeppool)))
        x_d2 = self.decoder_2(self.MFM2(x_e2) + self.up(x_d3) + self.up_8(self.deeppoolconv2(x_deeppool)))
        x_d1 = self.decoder_1(self.MFM1(x_e1) + self.up(x_d2) + self.up_16(self.deeppoolconv1(x_deeppool)))

        output = self.output_1(x_d1)
        output_deeppool = self.output_deeppool(x_deeppool)
        output_deeppool = F.interpolate(output_deeppool, scale_factor=16, mode='bilinear', align_corners=True)

        if self.Train:
            return [torch.sigmoid(output),torch.sigmoid(output_deeppool)]
        else:
            return torch.sigmoid(output)


if __name__ == '__main__':

    def measure_runtime(model, input_size=(1, 1, 256, 256), iterations=300):
        model.eval()
        input_data = torch.randn(input_size)
        if torch.cuda.is_available():
            model.cuda()
            input_data = input_data.cuda()

        with torch.no_grad():
            model(input_data)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model(input_data)
        end_time = time.time()

        avg_runtime = (end_time - start_time) / iterations * 1000
        return avg_runtime

    model = FSGNet(Train=False).cuda()
    x = torch.randn(1, 1, 256, 256).cuda()
    output = model(x)
    flops, params = profile(model, (x,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')

    runtime = measure_runtime(model)

    print(f"Model 平均运行时间: {runtime:.2f} ms")
