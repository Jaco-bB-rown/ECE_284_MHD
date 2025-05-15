from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms.v2 as v2
def dataTransforms(data_aug_type,size=(256,256), mask=True):
    """
    Perfroms the requested data augmentation
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    if mask:
        mask_transform = weights.transforms()
    else: #dont use it
        mask_transfrom = v2.Lambda(lambda y: y)
    match data_aug_type:
        case "1":
            return v2.Compose(
            [
            weights.transforms(),
            v2.Resize(size=size),
            ]
            )