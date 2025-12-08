"""Global configuration for robot navigation system"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ========== Robot Connection ==========
ROBOT_IP = os.getenv('ROBOT_IP', '192.168.2.2')
ROBOT_PORT = int(os.getenv('ROBOT_PORT', 31001))
CONNECTION_TIMEOUT = int(os.getenv('CONNECTION_TIMEOUT', 4))

# ========== Data Paths ==========
# Default data directory (images) — overridable via env
DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'perception' / 'data_samples'))
DEFAULT_IMAGE = Path(os.getenv('DEFAULT_IMAGE', DATA_DIR / 'example.png'))

# ========== Model Configuration ==========
# - 'unet': standard U-Net 
# - 'unetpp': U-Net++ 
# - 'attunet': Attention U-Net 
MODEL_DIR = Path(os.getenv('MODEL_DIR', PROJECT_ROOT / 'checkpoint'))
MODEL_TYPE = os.getenv('MODEL_TYPE', 'unetpp')  # unet | unetpp | attunet
MODEL_PATH = Path(os.getenv('MODEL_PATH', MODEL_DIR / 'unet++' / 'unet++_625.pth'))
DEVICE = 'cuda' if os.getenv('USE_CUDA', 'false').lower() == 'true' else 'cpu'


# ========== Image Processing ==========
INPUT_SIZE = (128, 128)
CAMERA_OUTPUT_SIZE = (640, 480)
MASK_THRESHOLD = float(os.getenv('MASK_THRESHOLD', 0.5))
MIN_AREA_SIZE = int(os.getenv('MIN_AREA_SIZE', 120))

# ========== Normalization ==========
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ========== Motion Control ==========
VELOCITY = 0.7
ANGLE_TOLERANCE = 60  # degrees
ANGLE_THRESHOLD = 50  # degrees
MAX_STOPFLAG = 5

# ========== Path Planning ==========
CLUSTER_LOOKAHEAD = 2.0
PATH_SMOOTHING_WINDOW = 5
EDGE_DETECTION_RANGE = 4

# ========== Visualization ==========
SHOW_DEBUG_PLOT = False
SHOW_MAP = True
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
