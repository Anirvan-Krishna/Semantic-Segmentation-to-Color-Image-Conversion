import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
learning_rate = 2e-4
batch_size = 8
n_workers = 2
channels_img = 3
L1_lambda = 100
n_epochs = 500
load_model = False
save_model = True
checkpoint_disc = "disc.path.tar"
checkpoint_gen = "gen.path.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5)], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

transform_only_target = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ]
)