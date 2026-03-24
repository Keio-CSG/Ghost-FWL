from dataclasses import dataclass, field

import yaml


@dataclass
class TrainingConfig:
    ###########################################################
    # training
    ###########################################################
    seed: int = 42
    lr: float = 0.0001
    batch_size: int = 8
    epochs: int = 10
    device: str = "cuda:0"
    ###########################################################
    # Input
    ###########################################################
    train_voxel_dirs: list[str] = field(default_factory=lambda: [])
    train_annotation_dirs: list[str] = field(default_factory=lambda: [])
    valid_voxel_dirs: list[str] = field(default_factory=lambda: [])
    valid_annotation_dirs: list[str] = field(default_factory=lambda: [])
    train_peak_dirs: list[str] = field(default_factory=lambda: [])
    train_annotation_peak_dirs: list[str] = field(default_factory=lambda: [])
    valid_peak_dirs: list[str] = field(default_factory=lambda: [])
    valid_annotation_peak_dirs: list[str] = field(default_factory=lambda: [])

    checkpoint_path: str = ""
    pretrained_model_path: str = ""
    freeze_encoder: bool = False  # Whether to freeze encoder during training
    ###########################################################
    # Output, log
    ###########################################################
    config_name: str = "train"
    save_model_dir: str = ""
    save_model_interval: int = 5
    is_train: bool = True
    is_log: bool = False
    log_interval: int = 100
    ###########################################################
    # dataset, dataloader
    ###########################################################
    num_workers: int = 4
    divide: int = 1
    ###########################################################
    # model
    ###########################################################
    model_name: str = ""
    num_classes: int = 4
    # UNet3D specific parameters
    n_channels: int = 1  # Number of input channels for UNet3D
    bilinear: bool = False  # Use bilinear interpolation for upsampling
    target_size: list[int] = field(
        default_factory=lambda: [64, 64, 400]
    )  # Target voxel size [X, Y, Z]
    downsample_z: int | None = None  # Downsample histogram direction (None means no downsampling)
    y_crop_top: int = 0  # Crop top of Y axis
    y_crop_bottom: int = 0  # Crop bottom of Y axis
    z_crop_front: int = 0  # Crop front of Z axis
    z_crop_back: int = 0  # Crop back of Z axis

    mask_ratio: float = 0.15  # Ratio of X,Y coordinates to mask
    mask_value: float = 0.0  # Value to use for masked positions
    max_peaks: int = 4

    voxel_size: list[int] = field(default_factory=lambda: [256, 128, 128])
    patch_size: list[int] = field(default_factory=lambda: [256, 16, 16])
    encoder_embed_dim: int = 768
    encoder_depth: int = 6
    encoder_num_heads: int = 6
    decoder_embed_dim: int = 384
    decoder_depth: int = 6
    decoder_num_heads: int = 6
    drop_rate: float = 0.2
    attn_drop_rate: float = 0.2
    drop_path_rate: float = 0.2
    mlp_ratio: float = 4.0
    ###########################################################
    # Labels to ignore during training and visualization
    ###########################################################
    ignore_visualize_labels: list[int] = field(
        default_factory=lambda: []
    )  # Labels to ignore in visualization
    ignore_train_labels: list[int] = field(
        default_factory=lambda: [-1]
    )  # Labels to ignore in training
    ###########################################################
    # loss, optimizer, scheduler
    ###########################################################
    loss_fn: str = "focal"

    position_weight: float = 1.0
    height_weight: float = 1.0
    width_weight: float = 0.5
    mae_reconstruction_weight: float = 1.0
    position_loss: str = "l1"
    height_loss: str = "l1"
    width_loss: str = "l1"
    mae_reconstruction_loss: str = "mse"

    focal_gamma: float = 2.0
    focal_alpha: list[float] = field(default_factory=lambda: [0.0001, 0.05, 0.25, 0.7])
    optimizer: str = "adamw"
    scheduler: str = ""


@dataclass
class TestConfig:
    ###########################################################
    # test
    ###########################################################
    seed: int = 42
    batch_size: int = 8
    device: str = "cuda:0"
    ###########################################################
    # Input
    ###########################################################
    checkpoint_path: str = ""
    # Voxel MAE reconstruction: voxel-only directories
    test_voxel_dirs: list[str] = field(default_factory=lambda: [])
    test_annotation_dirs: list[str] = field(default_factory=lambda: [])
    test_peak_dirs: list[str] = field(default_factory=lambda: [])
    test_annotation_peak_dirs: list[str] = field(default_factory=lambda: [])
    ###########################################################
    # Output, log
    ###########################################################
    config_name: str = "test"
    is_train: bool = False
    is_log: bool = True
    log_interval: int = 100
    output_dir: str | None = None
    ###########################################################
    # dataset, dataloader
    ###########################################################
    num_workers: int = 4
    divide: int = 1
    ###########################################################
    # model
    ###########################################################
    model_name: str = ""
    num_classes: int = 4
    n_channels: int = 1
    bilinear: bool = False
    target_size: list[int] = field(
        default_factory=lambda: [64, 64, 400]
    )  # Target voxel size (X, Y, Z)
    downsample_z: int | None = None  # Downsample histogram direction (None means no downsampling)
    y_crop_top: int = 0  # Crop top of Y axis
    y_crop_bottom: int = 0  # Crop bottom of Y axis
    z_crop_front: int = 0  # Crop front of Z axis
    z_crop_back: int = 0  # Crop back of Z axis
    mask_ratio: float = 0.15  # Ratio of X,Y coordinates to mask
    mask_value: float = 0.0  # Value to use for masked positions
    max_peaks: int = 4
    use_threshold_prediction: bool = True  # Enable threshold-based prediction
    prediction_threshold: float = 0.5  # Minimum probability threshold (0.0-1.0)

    voxel_size: list[int] = field(default_factory=lambda: [256, 128, 128])
    patch_size: list[int] = field(default_factory=lambda: [256, 16, 16])
    encoder_embed_dim: int = 768
    encoder_depth: int = 6
    encoder_num_heads: int = 6
    decoder_embed_dim: int = 384
    decoder_depth: int = 6
    decoder_num_heads: int = 6
    drop_rate: float = 0.2
    attn_drop_rate: float = 0.2
    drop_path_rate: float = 0.2
    mlp_ratio: float = 4.0
    ###########################################################
    # Labels to ignore during training and visualization
    ###########################################################
    ignore_visualize_labels: list[int] = field(
        default_factory=lambda: []
    )  # Labels to ignore in visualization
    ignore_train_labels: list[int] = field(
        default_factory=lambda: [-1]
    )  # Labels to ignore in training


def load_config_from_yaml(yaml_file: str) -> TrainingConfig | TestConfig:
    with open(yaml_file, "r") as file:
        config_data = yaml.safe_load(file)

    if config_data["config_name"] == "train":
        return TrainingConfig(**config_data)
    elif config_data["config_name"] == "test":
        return TestConfig(**config_data)
    else:
        raise ValueError(f"Invalid config name: {config_data['config_name']}")
