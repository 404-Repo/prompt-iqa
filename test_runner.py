import argparse
from PIL import Image
from time import time

import numpy as np
import torch
from loguru import logger

from PromptIQA.prompt_iqa_pipeline import PromptIQAPipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to the image that will be evaluated.")
    args = parser.parse_args()

    pil_image = Image.open(args.image)
    torch_image = torch.tensor(np.asarray(pil_image))

    prompt_iqa_pipe = PromptIQAPipeline()
    prompt_iqa_pipe.load_pipeline()

    t1 = time()
    score = prompt_iqa_pipe.compute_quality(torch_image)
    t2 = time()
    logger.info(f"Current score: {score}")
    logger.info(f"Image inference took: {t2 - t1} sec")
