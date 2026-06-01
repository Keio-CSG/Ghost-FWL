# Ghost-FWL Dataset

## Dataset Download
- [Download Link](https://drive.google.com/drive/folders/1C7jerNeQf2rsT53lUmA20YUb4PNCOnp4?usp=drive_link)

## Dataset Structure
- After installation, unzip and place the files in the following directory structure:

- Please delete the following files as the original data is corrupted:
    ```bash
    rm ghost_dataset/scene003/annotation_v1/hist022/20250929162519_t01759130735367000000_000043_annotation_voxel.b2
    rm ghost_dataset/scene003/annotation_v1_expand/hist022/20250929162519_t01759130735367000000_000043_annotation_voxel.b2
    rm ghost_dataset/scene003/data/hist022/20250929162519_t01759130735367000000_000043_voxel.b2
    ```

### Dataset for pretraining
```bash
mae_dataset/
├── ghost/
│   ├── YYYYMMDDHHMMSS_voxel_b2/
│   │   ├── YYYYMMDDHHMMSS_{t}_{id}_voxel.b2
│   │   └── ...
│   ├── YYYYMMDDHHMMSS_voxel_b2/
│   │   ├── YYYYMMDDHHMMSS_{t}_{id}_voxel.b2
│   │   └── ...
│   ├── ...
│   └── peaks/
│       ├── YYYYMMDDHHMMSS_voxel_b2/
│       │   ├── YYYYMMDDHHMMSS_{t}_{id}_peak.npy
│       │   └── ...
│       ├── YYYYMMDDHHMMSS_voxel_b2/
│       │   ├── YYYYMMDDHHMMSS_{t}_{id}_peak.npy
│       │   └── ...
│       └── ...
│
└── normal/
    ├── YYYYMMDDHHMMSS/
    │   ├── YYYYMMDDHHMMSS_{t}_{id}_voxel.b2
    │   └── ...
    ├── YYYYMMDDHHMMSS/
    │   ├── YYYYMMDDHHMMSS_{t}_{id}_voxel.b2
    │   └── ...
    ├── ...
    └── peaks/
       ├── YYYYMMDDHHMMSS/
       │   ├── YYYYMMDDHHMMSS_{t}_{id}_peak.npy
       │   └── ...
       ├── YYYYMMDDHHMMSS/
       │   ├── YYYYMMDDHHMMSS_{t}_{id}_peak.npy
       │   └── ...
       └── ...
```

- YYYYMMDDHHMMSS_{t}_{id}_voxel.b2: voxel grid file
    ```python
    from src.utils import load_blosc2
    voxel_grid = load_blosc2(voxel_file)
    print(voxel_grid.shape)
    ```
    ```bash
    (400, 512, 700)
    ```
- peaks/YYYYMMDDHHMMSS/YYYYMMDDHHMMSS_{t}_{id}_peak.npy: peak file
   ```bash
   import numpy as np
   peak_data = np.load(peak_file)
   print(peak_data.shape)
   ```
   ```bash
   (204800, 3) # 400 * 512 = 204800
   [x, y, [peak_position, peak_intensity, peak_width]]
   ```

### Annotation Dataset for Ghost Detection
```bash
ghost_dataset/
├── scene001/
│   ├── annotation_v{X}/
│   │   ├── hist001/
│   │   │   ├── YYYYMMDDHHMMSS_{t}_{id}_annotation_voxel.b2
│   │   │   └── ...
│   │   ├── hist002/
│   │   └── ...
│   ├── annotation_v{X}_expand/
│   │   ├── hist001/
│   │   │   ├── YYYYMMDDHHMMSS_{t}_{id}_annotation_voxel.b2
│   │   │   └── ...
│   │   ├── hist002/
│   │   └── ...
│   └── data/
│       ├── hist001/
│       │   ├── YYYYMMDDHHMMSS_{t}_{id}_annotation_voxel.b2
│       │   └── ...
│       ├── hist002/
│       └── ...
│
├── scene002/
│   ├── annotation_v{X}/
│   ├── annotation_v{X}_expand/
│   └── data/
│
└── ...
```

- YYYYMMDDHHMMSS_{t}_{id}_annotation_voxel.b2: annotation voxel file
    ```bash
    (400, 512, 700)
    ```
- YYYYMMDDHHMMSS_{t}_{id}_voxel.b2: voxel file
    ```bash
    (400, 512, 700)
    ```
