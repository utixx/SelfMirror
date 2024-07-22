import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TwoConv, self).__init__()

        # self.out_channels = out_channels
        # print(out_channels//2)

        self.two_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2,
                      kernel_size=kernel_size, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels,
                      kernel_size=kernel_size, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.two_conv(x)


class Enconder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, is_downsample=True):
        super(Enconder, self).__init__()

        self.biconv =TwoConv(in_channels, out_channels, kernel_size)

        if is_downsample:
            self.down = nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                                  kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            self.down = None

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        enc_fm = self.biconv(x)

        return enc_fm


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1), output_padding=1)

        self.biconv = TwoConv(2 * out_channels, out_channels, kernel_size)

    def forward(self, x, encoder_features):
        x_up = self.up(x)
        # print(x_up.shape, encoder_features.shape)

        x_up = torch.cat((x_up, encoder_features), dim=1)

        # print(x_up.shape)

        x_up = self.biconv(x_up)

        return x_up


class UNet3D(nn.Module):
    def __init__(self, in_channels, f_maps=[16, 32, 64], final_sigmoid_for_test=False):
        super(UNet3D, self).__init__()

        self.f_maps = f_maps
        self.enc = self.squeeze(f_maps=[in_channels] + f_maps)
        self.dec = self.expend(f_maps=f_maps[::-1] + [in_channels])

        self.final_conv = nn.Conv3d(f_maps[0], in_channels, 1)

        if final_sigmoid_for_test:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

    def squeeze(self, f_maps):
        model_list = nn.ModuleList([])
        for i in range(1, len(f_maps)):
            if i == 1:
                enc_i = Enconder(in_channels=f_maps[i - 1], out_channels=f_maps[i], is_downsample=False)
                model_list.append(enc_i)
            else:
                enc_i = Enconder(in_channels=f_maps[i - 1], out_channels=f_maps[i])
                model_list.append(enc_i)
        return model_list

    def expend(self, f_maps):
        model_list = nn.ModuleList([])
        for i in range(1, len(f_maps)):
            if i == len(f_maps) - 1:
                decoder_i = Decoder(in_channels=f_maps[i - 1], out_channels=f_maps[i - 1])
                model_list.append(decoder_i)
            else:
                decoder_i = Decoder(in_channels=f_maps[i - 1], out_channels=f_maps[i])
                model_list.append(decoder_i)
        return model_list

    def forward(self, x):
        enc_list_for_cat = []

        for enc_i in self.enc:
            x = enc_i(x)
            enc_list_for_cat.insert(0, x)

        enc_list_for_cat = enc_list_for_cat[1:]

        for dec_i, enc_list_for_cat_i in zip(self.dec, enc_list_for_cat):
            x = dec_i(x, enc_list_for_cat_i)

        x = self.final_conv(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


if __name__ == '__main__':
    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet3D(in_channels=1, f_maps=[16, 32, 64, 128]).to(device)

    summary(model, (1, 1, 64, 64, 64))
