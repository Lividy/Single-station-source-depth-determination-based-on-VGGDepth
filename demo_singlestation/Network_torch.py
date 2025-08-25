import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):  ##for location
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d((1, 4))
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d((1, 4))
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d((1, 2))
        
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.drop5 = nn.Dropout2d(p=0.5)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv11 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=(1,2), stride=(1, 2))
        self.conv13 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=(8,3), stride=(8, 3))
        self.conv15 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv16 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv17 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv18 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv19 = nn.Conv2d(32, out_channels, 3, padding=1)
    
        

    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.conv1(x))   # ,12,1024
        conv1 = F.relu(self.conv2(conv1)) # 12,1024
        pool1 = self.pool1(conv1) # 12,256
        
        conv2 = F.relu(self.conv3(pool1))   # 12,256
        conv2 = F.relu(self.conv4(conv2))  # 12,256
        pool2 = self.pool2(conv2)         # 12,64
        
        conv3 = F.relu(self.conv5(pool2))  # 12,64
        conv3 = F.relu(self.conv6(conv3))  # 12,64
        pool3 = self.pool3(conv3)      # 12,32
        
        conv4 = F.relu(self.conv7(pool3))  # 12,32
        conv4 = F.relu(self.conv8(conv4))  # 12,32
        drop4 = self.drop4(conv4)    # 12 32    
        pool4 = self.pool4(drop4)   # 6,16
        
        conv5 = F.relu(self.conv9(pool4)) # 6,16
        conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16  
        drop5 = self.drop5(conv5)  # 6,16 
        
        # Decoder
        up6 = self.up6(drop5)  # 12,32
        up6 = torch.cat((up6, drop4), dim=1)  
        conv6 = F.relu(self.conv11(up6)) # 12,32
        conv6 = F.relu(self.conv12(conv6)) # 12,32
        #print(conv6.shape)

        up7 = self.up7(conv6)  # 12,64
        #print(up7.shape)
        up7 = torch.cat((up7, conv3), dim=1)  # 12,64
        #print(up7.shape)
        conv7 = F.relu(self.conv13(up7)) # 12,64
        conv7 = F.relu(self.conv14(conv7)) #    12,64
        #print("conv7", conv7.shape)
        up8 = self.up8(conv7)  # 96，192
        #print("up8", up8.shape)
        conv8 = F.relu(self.conv15(up8)) # 96,192
        conv8 = F.relu(self.conv16(conv8)) 
        conv8 = F.relu(self.conv17(conv8))
        #print("conv8", conv8.shape)
        conv9 = F.relu(self.conv18(conv8))
        #print("conv9", conv9.shape)
        conv10 = torch.sigmoid(self.conv19(conv9))
        #print("conv10", conv10.shape)
        
        return conv10


class UNet_d(nn.Module):  ##for location
    def __init__(self, in_channels, out_channels):
        super(UNet_d, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d((1, 4))
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d((1, 4))
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d((1, 2))
        
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.drop5 = nn.Dropout2d(p=0.5)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv11 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=(1,2), stride=(1, 2))
        self.conv13 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=(8,1), stride=(8, 1))
        self.conv15 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv16 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv17 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv18 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv19 = nn.Conv2d(32, out_channels, 3, padding=1)
    
        

    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.conv1(x))   # ,12,1024
        conv1 = F.relu(self.conv2(conv1)) # 12,1024
        pool1 = self.pool1(conv1) # 12,256
        
        conv2 = F.relu(self.conv3(pool1))   # 12,256
        conv2 = F.relu(self.conv4(conv2))  # 12,256
        pool2 = self.pool2(conv2)         # 12,64
        
        conv3 = F.relu(self.conv5(pool2))  # 12,64
        conv3 = F.relu(self.conv6(conv3))  # 12,64
        pool3 = self.pool3(conv3)      # 12,32
        
        conv4 = F.relu(self.conv7(pool3))  # 12,32
        conv4 = F.relu(self.conv8(conv4))  # 12,32
        drop4 = self.drop4(conv4)    # 12 32    
        pool4 = self.pool4(drop4)   # 6,16
        
        conv5 = F.relu(self.conv9(pool4)) # 6,16
        conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16  
        drop5 = self.drop5(conv5)  # 6,16 
        
        # Decoder
        up6 = self.up6(drop5)  # 12,32
        up6 = torch.cat((up6, drop4), dim=1)  
        conv6 = F.relu(self.conv11(up6)) # 12,32
        conv6 = F.relu(self.conv12(conv6)) # 12,32
        #print(conv6.shape)

        up7 = self.up7(conv6)  # 12,64
        #print(up7.shape)
        up7 = torch.cat((up7, conv3), dim=1)  # 12,64
        #print(up7.shape)
        conv7 = F.relu(self.conv13(up7)) # 12,64
        conv7 = F.relu(self.conv14(conv7)) #    12,64
        #print("conv7", conv7.shape)
        up8 = self.up8(conv7)  # 96，192
        #print("up8", up8.shape)
        conv8 = F.relu(self.conv15(up8)) # 96,192
        conv8 = F.relu(self.conv16(conv8)) 
        conv8 = F.relu(self.conv17(conv8))
        #print("conv8", conv8.shape)
        conv9 = F.relu(self.conv18(conv8))
        #print("conv9", conv9.shape)
        conv10 = torch.sigmoid(self.conv19(conv9))
        #print("conv10", conv10.shape)
        
        return conv10








class UNetDepth(nn.Module):  ##for location
    def __init__(self, in_channels, out_channels):
        super(UNetDepth, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d((2, 4))
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d((2, 4))
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d((3, 2))
        
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.pool4 = nn.MaxPool2d((1,2))
        
        self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.drop5 = nn.Dropout2d(p=0.5)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=(1,2), stride=(1, 2))
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=(1,2), stride=(1, 2))
        self.conv13 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=(1,2), stride=(1, 2))
        self.conv15 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv16 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv17 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv18 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv19 = nn.Conv2d(32, 1, 3, padding=1)
        #self.fc = nn.Linear(256, 64)

    
        

    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.conv1(x))   # ,12,1024
        conv1 = F.relu(self.conv2(conv1)) # 12,1024
        pool1 = self.pool1(conv1) # 12,256
        
        conv2 = F.relu(self.conv3(pool1))   # 12,256
        conv2 = F.relu(self.conv4(conv2))  # 12,256
        pool2 = self.pool2(conv2)         # 12,64
        
        conv3 = F.relu(self.conv5(pool2))  # 12,64
        conv3 = F.relu(self.conv6(conv3))  # 12,64
        pool3 = self.pool3(conv3)      # 12,32
        
        conv4 = F.relu(self.conv7(pool3))  # 12,32
        conv4 = F.relu(self.conv8(conv4))  # 12,32
        drop4 = self.drop4(conv4)    # 12 32    
        pool4 = self.pool4(drop4)   # 6,16
        
        conv5 = F.relu(self.conv9(pool4)) # 6,16
        conv5 = F.relu(self.conv10(conv5)) # 6,16
        conv5 = F.relu(self.conv10(conv5)) # 6,16
        conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16
        # conv5 = F.relu(self.conv10(conv5)) # 6,16  
        drop5 = self.drop5(conv5)  # 6,16 
        
        # Decoder
        up6 = self.up6(drop5)  # 12,32
        conv6 = F.relu(self.conv11(up6)) # 12,32
        conv6 = F.relu(self.conv12(conv6)) # 12,32
        #print(conv6.shape)

        up7 = self.up7(conv6)  # 12,64

        print(up7.shape)
        conv7 = F.relu(self.conv13(up7)) # 12,64
        conv7 = F.relu(self.conv14(conv7)) #    12,64
        print("conv7", conv7.shape)
        up8 = self.up8(conv7)  # 96，192
        print("up8", up8.shape)
        conv8 = F.relu(self.conv15(up8)) # 96,192
        conv8 = F.relu(self.conv16(conv8)) 
        conv8 = F.relu(self.conv17(conv8))
        print("conv8", conv8.shape)
        conv9 = F.relu(self.conv18(conv8))
        conv9 = F.relu(self.conv19(conv9))
        print("conv9", conv9.shape)
        fc10 = self.fc(conv9)
        fc10 = torch.sigmoid(fc10)
        print("fc10", fc10.shape)
        #fc10 = F.softmax(fc10, dim=1)
        #print("fc10", fc10.shape)
        
        return fc10


class VGGDepth(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepth, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,64)
        )

    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x

class VGG19Depth(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG19Depth, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,64)
        )

    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x


 
# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out
 
# 定义VGG16-attention模型
class VGG16Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG16Attention, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SelfAttention(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),


            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SelfAttention(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),


            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),


            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),


            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 64)
        )
 
    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x
 









class VGGDepth1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepth1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,4)),

            nn.Conv2d(512, 1024,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=4)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,64)
        )

    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x

# class VGGDepth1024_1(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(VGGDepth1024_1, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((1,4)),
            
            

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((1,4)),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((4,4)),

#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((3,4)),

#             nn.Conv2d(512, 1024,kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=1, stride=4)
            
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(1024 * 1 * 1, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(2048, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(1024,1024)
#         )

#     def forward(self, x):
#         x = self.features(x)
        
        
#         #print(x.shape)
#         x = torch.flatten(x, 1)
#         #x = x.view(x.size(0), -1)
#         #print(x.shape)
#         x = self.classifier(x)
#         x= torch.sigmoid(x)

#         return x

class VGGDepth1024(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepth1024, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,1024)
        )
    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x

class VGGDepth256(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepth256, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,256)
        )
    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x

import torch
import torch.nn as nn


class VGGDepth256_epi_xy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepth256_epi_xy, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 1))

        )

        self.classifier = nn.Sequential(
            nn.Linear(16*(16+3), 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024 , 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256)


        )


    def forward(self, x, epicenter_distance, Lon, Lat):
        x = self.features(x)
        epicenter_distance = epicenter_distance.unsqueeze(1)
        Lon = Lon.unsqueeze(1)
        Lat = Lat.unsqueeze(1)
        epicenter_distance = epicenter_distance.unsqueeze(3)
        Lon = Lon.unsqueeze(3)
        Lat = Lat.unsqueeze(3)
        # 拼接 epicenter_distance
        x = torch.cat((x, epicenter_distance), dim=3)
        x = torch.cat((x, Lon), dim=3)
        x = torch.cat((x, Lat), dim=3)


        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x

    


class VGGDepthSingle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepthSingle, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,64)
        )

    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x

class VGGDepthSingle256(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepthSingle256, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,256)
        )

    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x

class VGGDepthSingle256_epi(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGDepthSingle256_epi, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),
            
            

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1,4)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,256)
        )

    def forward(self, x):
        x = self.features(x)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x= torch.sigmoid(x)

        return x