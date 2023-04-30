import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RoofCNN(nn.Module):
    """Simple CNN for Roof saw detection of Roof condition."""
    supported_backbones = ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
                            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
                            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                            'densenet121', 'densenet161', 'densenet169', 'densenet201',
                            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

    def __init__(self, num_classes, backbone='mobilenet_v2'):
        """_summary_

        Args:
            num_classes (_type_): _description_
            backbone (str, optional): _description_. Defaults to 'mobilenet_v2'.
        """
        # More flexible number of classes
        
        #  Add backbone to assertion: mobilene_v2, mobilenet_v3_small, mobilenet_v3_large
        #  resnet18, resnet34, resnet50, resnet101, resnet152, 
        #  efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
        #  efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
        #  densenet121, densenet161, densenet169, densenet201
        #  vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
        assert backbone in RoofCNN.supported_backbones
        
        super(RoofCNN, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Backbone MobileNet pretrained on ImageNet
        # without the last fully connected layer
        # Add other backbone
        # Resnet, EfficientNet, etc.   
        if "resnet" in backbone:
            resnet = getattr(models, backbone)(pretrained=True)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.classifier = nn.Linear(resnet.fc.in_features, num_classes)
        elif "mobilenet_v2" in backbone:
            mobilenet_v2 = getattr(models, backbone)(pretrained=True)
            self.encoder = nn.Sequential(*list(mobilenet_v2.children())[:-1])
            self.classifier = nn.Linear(1280, num_classes)
        elif "mobilenet_v3" in backbone:
            mobilenet_v3 = getattr(models, backbone)(pretrained=True)
            self.encoder = nn.Sequential(*list(mobilenet_v3.children())[:-1])
            self.classifier = nn.Linear(576 if 'small' in backbone else 960, num_classes)
        elif "efficientnet" in backbone:
            url = 'rwightman/gen-efficientnet-pytorch'
            efficientnet = torch.hub.load(url, backbone, pretrained=True)
            self.encoder = nn.Sequential(*list(efficientnet.children())[:-1])
            self.classifier = nn.Linear(efficientnet.classifier.in_features, num_classes)
        elif "densenet" in backbone:
            densenet = getattr(models, backbone)(pretrained=True)
            self.encoder = nn.Sequential(*list(densenet.children())[:-1])
            self.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
        elif "vgg" in backbone:
            vgg = getattr(models, backbone)(pretrained=True)
            self.encoder = nn.Sequential(*list(vgg.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            raise ValueError(f"{backbone} is not a valid model")
    
    # TODO If add other backbone need to reconsider this
    def _forward_impl(self, x):
        # Feature extraction
        x = self.encoder(x)
        
        # Transform before classifier
        # Mobilenet v2 and v3 small
        if self.backbone in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
            # Cannot use "squeeze" as batch-size can be 1
            # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            # x = torch.flatten(x, 1)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        # For Resnet
        elif 'resnet' in self.backbone:
            x = x.view(x.size(0), -1)
        # For EfficientNet
        elif 'efficientnet' in self.backbone:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        elif 'densenet' in self.backbone:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif 'vgg' in self.backbone:
            x = F.adaptive_avg_pool2d(x, (7, 7))
            x = x.view(x.size(0), -1)
            
        # FC layer
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)