import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.down1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.down2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')

        self.down3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.down4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')

        self.down5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.down6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.down7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.down8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same')

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same')

        self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same')

        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')

        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')

        self.finalconv = nn.Conv2d(in_channels=32, out_channels=19, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.down1(x)
        x = self.relu(x)
        x = self.down2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.down3(x)
        x = self.relu(x)
        x = self.down4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.down5(x)
        x = self.relu(x)
        x = self.down6(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.down7(x)
        x = self.relu(x)
        x = self.down8(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.upconv1(x)
        x = self.up1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.relu(x)

        x = self.upconv2(x)
        x = self.up2(x)
        x = self.relu(x)
        x = self.up2(x)
        x = self.relu(x)

        x = self.upconv3(x)
        x = self.up3(x)
        x = self.relu(x)
        x = self.up3(x)
        x = self.relu(x)

        x = self.upconv4(x)
        x = self.up4(x)
        x = self.relu(x)
        x = self.up4(x)
        x = self.relu(x)
        
        x = self.finalconv(x)

        return x