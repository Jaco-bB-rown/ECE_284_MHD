from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import ResNet152_Weights
import torchvision.transforms.v2 as v2
def dataTransforms(data_aug_type,size=(256,256), mask=True):
    """
    data_aug_type: Perfroms the requested data augmentation
    size: The output size of the image
    mask: Whether we are training the mask model
    """
    mask_weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    class_weights = ResNet152_Weights.DEFAULT
    if mask:
        model_transform = v2.Compose(
            [mask_weights.transforms(),
             v2.Resize(size=size),
            ])
    else: 
        model_transform = class_weights.transforms()
    match data_aug_type:
        case "1":
            return v2.Compose(
            [
            model_transform,
    
            ]
            )
        case "2":
            return v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=1.0),
                    v2.RandomVerticalFlip(p=1.0),
                ]
            )
        case "3":
            return v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=1.0),
                ]
            )
        case "4":
            return v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=1.0),
                ]
            )