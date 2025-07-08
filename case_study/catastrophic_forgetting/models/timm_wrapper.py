import timm
import torch.nn as nn


def vit_base_patch16_224():
    net = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=101, in_chans=3)
    net = ViTTimmModelWrapper(net)
    return net

def resnet50():
    net = timm.create_model("resnet50", pretrained=True, num_classes=101, in_chans=3)
    net = ResNetTimmModelWrapper(net)
    return net
     

class ViTTimmModelWrapper(nn.Module):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
        # for param in self.vit.parameters():
        #     param.requires_grad = False

        # for param in self.vit.head.parameters():
        #     param.requires_grad = True
    
    def forward(self, x):
        return self.net(x)
    
    def feature(self, x):
        x = self.net.forward_features(x)
        if self.net.attn_pool is not None:
            x = self.net.attn_pool(x)
        elif self.net.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.net.global_pool:
            x = x[:, 0]  # class token
        x = self.net.fc_norm(x)
        x = self.net.head_drop(x)
        return x
    
    def prediction(self, x, pre_logits=False):
        return x if pre_logits else self.net.head(x)
    
class ResNetTimmModelWrapper(nn.Module):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

        # for param in self.net.parameters():
        #     param.requires_grad = False

        # for param in self.net.fc.parameters():
        #     param.requires_grad = True
    
    def forward(self, x):
        return self.net(x)
    
    def feature(self, x):
        x = self.net.forward_features(x)
        x = self.net.global_pool(x)
        # if self.drop_rate:
        #     x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x
    
    def prediction(self, x, pre_logits=False):
        return x if pre_logits else self.net.fc(x)



if __name__ == "__main__":
    import torch
    x = torch.randn((10, 3, 224, 224))
    # net = vit_base_patch16_224()
    net = resnet50()
    # print("net\n", net)
    # print("features\n", net.net.forward_features)
    # print("classifiers\n", net.net.get_classifier())
    
    pred_out = net(x)
    print(f"model has output shape {pred_out.shape}")
    features = net.feature(x)
    print(f"model has feature shape {features.shape}")
    out = net.prediction(features)
    print(f"model has output shape {out.shape}")

