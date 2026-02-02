# config.py

DEVICE = "cuda"

# --- YOLO sensor gate ---
YOLO_CONF_TH = 0.4

# --- Geometry ---
MIN_ROI_SIZE = 32

# --- FAISS ---
FAISS_TOP_K = 3
RES_CONF_TH = 0.85

FAISS_INDEX_PATH = "models/chip.index"
FAISS_PATHS_PATH = "models/chip_paths.npy"

# --- Debug / Logs ---
LOG_EVERY = True
SAVE_DEBUG = False
DEBUG_DIR = "debug_rois"


# --- VLM ---
REFERENCE_DIR = "./chip_images/"