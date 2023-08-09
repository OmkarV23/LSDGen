import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from llm_descriptor.global_descriptor import GlobalDescriptor
from llm_descriptor.local_descriptor import LocalDescriptor
from llm_descriptor.visual_descriptor import VisualDescriptor
from llm_evaluator.evaluation_instruction import EvaluationInstructor

sys.path.insert(0, 'submodule/CenterNet2')
sys.path.insert(0, 'submodule/detectron2')
sys.path.insert(0, 'submodule/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import VisualizationDemo
from icecream import ic
from PIL import Image
WINDOW_NAME = "LLMScore(BLIPv2+GRiT+GPT-4)"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin llm_descriptor
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    parser.add_argument(
        "--piepline-cfg",
        default="/home/ovengurl/LSDGen/LLMScore/config.json",
        metavar="FILE",
        help="path to config file",
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()

    # read josn file
    with open(args.piepline_cfg) as f:
        config = json.load(f)
        args.config_file = config['detectron2']['config_file']
        args.opts = ["MODEL.WEIGHTS", config['detectron2']['model_path']]
        args.confidence_threshold = config['detectron2']['confidence-threshold']
        args.llm_id = config['llm_id']
        args.test_task = "DenseCap"
        args.cpu = False

    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    global_descriptor = GlobalDescriptor()
    local_descriptor = LocalDescriptor()
    llm_descriptor = VisualDescriptor(config["openai_key"], args.llm_id)
    llm_evaluator = EvaluationInstructor(config["openai_key"], args.llm_id)
    
    images = config['data']['images']
    text_prompts = config['data']['text_prompts']
    output_dir = config['output_dir']

    for img_src, text_prompt in zip(images, text_prompts):
    
        img = read_image(img_src, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                img_src,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        local_description = local_descriptor.dense_pred_to_caption(predictions)
        out_filename = os.path.join(output_dir, os.path.basename(img_src))
        visualized_output.save(out_filename)
        global_description = global_descriptor.get_global_description(img_src)
        image = Image.open(img_src)
        width, height = image.size
        scene_description = llm_descriptor.generate_multi_granualrity_description(global_description, local_description, width, height)
        ic(scene_description)
        overall, error_counting, overall_rationale, error_counting_rationale = llm_evaluator.generate_score_with_rationale(scene_description, text_prompt)
        ic(overall, overall_rationale)
        ic(error_counting, error_counting_rationale)