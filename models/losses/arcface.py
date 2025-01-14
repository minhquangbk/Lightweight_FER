import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5, criterion=nn.CrossEntropyLoss()):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.criterion = criterion

    def forward(self, cos_theta, labels):
        # Ensure cos_theta is clamped to avoid numerical issues with acos
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Calculate theta (angle) from cos_theta
        theta = torch.acos(cos_theta)
        
        # Add margin to the angles corresponding to the correct classes
        one_hot = torch.nn.functional.one_hot(labels, num_classes=cos_theta.size(1)).to(cos_theta.dtype)
        target_logits = torch.cos(theta + self.margin)
        
        # Scales up logits before applying softmax
        logits = cos_theta * (1 - one_hot) + target_logits * one_hot
        logits *= self.scale
        
        return self.criterion(logits, labels)

# Example usage:
if __name__ == "__main__":
    # Assume logits is the cosine of the angles between embeddings and class centers
    # logits should be of shape [batch_size, num_classes]
    logits = torch.randn(10, 5)  # Example logits
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # Corresponding labels

    loss_fn = ArcFaceLoss()
    loss = loss_fn(logits, labels)
    print("ArcFace Loss:", loss.item())
