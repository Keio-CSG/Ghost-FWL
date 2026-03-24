import json
from typing import Dict, List, Union
from .pointcloud_wrapper import PointcloudWrapper
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

import pandas as pd
import numpy as np
import json
from matplotlib.path import Path as PolygonPath
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull


class AnnotationLabel:
    NOISE = 0
    OBJECT = 1
    GLASS = 2
    GHOST = 3
    UNDEFINED = -1
