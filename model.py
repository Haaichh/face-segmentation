import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, features=32):
        super(UNet, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.down1 = nn.Conv2d(in_channels=3, out_channels=features, kernel_size=3, stride=1, padding='same')
        self.down2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding='same')

        self.down3 = nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=3, stride=1, padding='same')
        self.down4 = nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, stride=1, padding='same')

        self.down5 = nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=3, stride=1, padding='same')
        self.down6 = nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, stride=1, padding='same')

        self.down7 = nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=3, stride=1, padding='same')
        self.down8 = nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, stride=1, padding='same')

        self.conv1 = nn.Conv2d(in_channels=features*8, out_channels=features*16, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=features*16, out_channels=features*16, kernel_size=3, stride=1, padding='same')

        self.upconv1 = nn.ConvTranspose2d(in_channels=features*16, out_channels=features*8, kernel_size=2, stride=2)
        self.up1 = nn.Conv2d(in_channels=(features*8)*2, out_channels=features*8, kernel_size=3, stride=1, padding='same')
        self.up2 = nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, stride=1, padding='same')

        self.upconv2 = nn.ConvTranspose2d(in_channels=features*8, out_channels=features*4, kernel_size=2, stride=2)
        self.up3 = nn.Conv2d(in_channels=(features*4)*2, out_channels=features*4, kernel_size=3, stride=1, padding='same')
        self.up4 = nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, stride=1, padding='same')

        self.upconv3 = nn.ConvTranspose2d(in_channels=features*4, out_channels=features*2, kernel_size=2, stride=2)
        self.up5 = nn.Conv2d(in_channels=(features*2)*2, out_channels=features*2, kernel_size=3, stride=1, padding='same')
        self.up6 = nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, stride=1, padding='same')

        self.upconv4 = nn.ConvTranspose2d(in_channels=features*2, out_channels=features, kernel_size=2, stride=2)
        self.up7 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=3, stride=1, padding='same')
        self.up8 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding='same')

        self.finalconv = nn.Conv2d(in_channels=features, out_channels=19, kernel_size=1, stride=1)

    def forward(self, x):

        x0 = x
        x1 = self.down1(x)
        x1 = self.relu(x1)
        x1 = self.down2(x1)
        x1 = self.relu(x1)
        c1 = self.maxpool(x1)

        x2 = self.down3(c1)
        x2 = self.relu(x2)
        x2 = self.down4(x2)
        x2 = self.relu(x2)
        c2 = self.maxpool(x2)

        x3 = self.down5(c2)
        x3 = self.relu(x3)
        x3 = self.down6(x3)
        x3 = self.relu(x3)
        c3 = self.maxpool(x3)

        x4 = self.down7(c3)
        x4 = self.relu(x4)
        x4 = self.down8(x4)
        x4 = self.relu(x4)
        c4 = self.maxpool(x4)

        x5 = self.conv1(c4)
        x5 = self.relu(x5)
        x5 = self.conv2(x5)
        x5 = self.relu(x5)

        x6 = self.upconv1(x5)
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.up1(x6)
        x6 = self.relu(x6)
        x6 = self.up2(x6)
        x6 = self.relu(x6)

        x7 = self.upconv2(x6)
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.up3(x7)
        x7 = self.relu(x7)
        x7 = self.up4(x7)
        x7 = self.relu(x7)

        x8 = self.upconv3(x7)
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.up5(x8)
        x8 = self.relu(x8)
        x8 = self.up6(x8)
        x8 = self.relu(x8)

        x9 = self.upconv4(x8)
        x9 = torch.cat((x9, x1), dim=1)
        x9 = self.up7(x9)
        x9 = self.relu(x9)
        x9 = self.up8(x9)
        x9 = self.relu(x9)
        
        x = self.finalconv(x9)

        return x