# Ghost-FWL Dataset

## Dataset Structure
### Dataset for pretraining
```bash
Pretraining/
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

## Dataset Download
- [Download Link]()
