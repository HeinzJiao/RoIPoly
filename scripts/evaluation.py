"""
This script is used for evaluating predicted building polygons with various metrics including COCO IoU, Boundary IoU,
PoLiS, Maximum Tangent Angle Error, and Complex IoU (cIoU).

Usage:
    python eval_metrics.py --gt-file <path_to_ground_truth_annotation> --dt-file <path_to_prediction_file> --eval-type <evaluation_metric>

Arguments:
    --gt-file: Path to the COCO-format ground truth annotation file.
    --dt-file: Path to the COCO-format detection result (predictions) file.
    --eval-type: Type of evaluation metric to use. Choose from:
                 - "coco_iou"      : Standard COCO IoU evaluation.
                 - "boundary_iou"  : Boundary IoU evaluation with boundary dilation.
                 - "polis"         : PoLiS evaluation.
                 - "angle"         : Max tangent angle error evaluation.
                 - "ciou"          : Complex IoU evaluation.

Example:
    python eval_metrics.py --gt-file ./data/ground_truth.json --dt-file ./data/predictions.json --eval-type coco_iou
"""
import argparse
import sys
sys.path.append("..")
from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from hisup.utils.metrics.polis import PolisEval
from hisup.utils.metrics.angle_eval import ContourEval
from hisup.utils.metrics.cIoU import compute_IoU_cIoU

def coco_eval(annFile, resFile):
    type=1
    annType = ['bbox', 'segm']
    print('Running demo for *%s* results.' % (annType[type]))

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [100]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats

def boundary_eval(annFile, resFile):
    dilation_ratio = 0.02 # default settings 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    polisEval.evaluate()

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool)
    print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    parser.add_argument("--eval-type", default="coco_iou", choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"])
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    if eval_type == 'coco_iou':
        coco_eval(gt_file, dt_file)
    elif eval_type == 'boundary_iou':
        boundary_eval(gt_file, dt_file)
    elif eval_type == 'polis':
        polis_eval(gt_file, dt_file)
    elif eval_type == 'angle':
        max_angle_error_eval(gt_file, dt_file)
    elif eval_type == 'ciou':
        compute_IoU_cIoU(dt_file, gt_file)
    else:
        raise RuntimeError('please choose a correct type from \
                            ["coco_iou", "boundary_iou", "polis", "angle", "ciou"]')
