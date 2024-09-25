from detectron2.config import CfgNode as CN

def add_roipoly_config(cfg):
    """
    Add config for RoIPoly.
    """
    cfg.MODEL.RoIPoly = CN()
    cfg.MODEL.RoIPoly.NUM_CLASSES = 1  # valid or invalid corner
    cfg.MODEL.RoIPoly.NUM_PROPOSALS = 34 # number of proposal polygons per image; needs manual adjustment based on the specific dataset
    cfg.MODEL.RoIPoly.NUM_CORNERS = 30  # number of vertices per polygon; needs manual adjustment based on the specific dataset
    # RCNN Head.
    cfg.MODEL.RoIPoly.NHEADS = 8  # multi-head self-attention
    cfg.MODEL.RoIPoly.DROPOUT = 0.0
    cfg.MODEL.RoIPoly.DIM_FEEDFORWARD = 2048
    cfg.MODEL.RoIPoly.ACTIVATION = 'relu'
    cfg.MODEL.RoIPoly.HIDDEN_DIM = 256
    cfg.MODEL.RoIPoly.NUM_CLS = 1
    cfg.MODEL.RoIPoly.NUM_REG = 3
    cfg.MODEL.RoIPoly.NUM_COR = 3
    cfg.MODEL.RoIPoly.NUM_POL = 3
    cfg.MODEL.RoIPoly.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.RoIPoly.NUM_DYNAMIC = 2
    cfg.MODEL.RoIPoly.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.RoIPoly.CLASS_WEIGHT = 2.0
    cfg.MODEL.RoIPoly.GIOU_WEIGHT = 2.0
    cfg.MODEL.RoIPoly.L1_WEIGHT = 5.0
    cfg.MODEL.RoIPoly.DEEP_SUPERVISION = False
    cfg.MODEL.RoIPoly.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.RoIPoly.USE_FOCAL = True
    cfg.MODEL.RoIPoly.ALPHA = 0.25
    cfg.MODEL.RoIPoly.GAMMA = 2.0
    cfg.MODEL.RoIPoly.PRIOR_PROB = 0.01

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])

