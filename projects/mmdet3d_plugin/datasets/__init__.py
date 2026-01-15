from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

from .nuscenes_map_dataset import CustomNuScenesLocalMapDataset

# 条件导入 AV2，避免 numpy.typing 依赖问题
try:
    from .av2_map_dataset import CustomAV2LocalMapDataset
    __all__ = [
        'CustomNuScenesDataset','CustomNuScenesLocalMapDataset', 'CustomAV2LocalMapDataset'
    ]
except ImportError:
    __all__ = [
        'CustomNuScenesDataset','CustomNuScenesLocalMapDataset'
    ]
