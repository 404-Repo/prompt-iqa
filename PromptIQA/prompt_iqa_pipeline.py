import os
import yaml
import random
import re
from pathlib import Path

import torchvision
import cv2
import torch
from PromptIQA.models import promptiqa
import numpy as np
from PromptIQA.utils.dataset.process import ToTensor, Normalize
from PromptIQA.utils.toolkit import *
import warnings
from time import time

warnings.filterwarnings("ignore")


class PromptIQAPipeline:
    def __init__(self):
        self._model: promptiqa.PromptIQA = None
        self._config_data: dict = None
        self.transform = torchvision.transforms.Compose(
            [Normalize(0.5, 0.5), ToTensor()]
        )

    def _load_config(self) -> None:
        """ Function for loading the data from the .yaml configuration file. """

        with open("./prompt_iqa_conf.yml", "r") as file:
            self._config_data = yaml.safe_load(file)
        assert self._config_data != {}

    def load_pipeline(self) -> None:
        self._load_config()
        self.load_model(self._config_data["checkpoint_path"])
        self.load_ref_dataset(self._config_data["reference_dataset_path"], self._config_data["reference_weights"])

    def load_model(self, ckpt_path: str) -> None:
        self._model = promptiqa.PromptIQA()

        dict_pkl = {}
        for key, value in torch.load(ckpt_path, map_location="cpu")["state_dict"].items():
            dict_pkl[key[7:]] = value
        self._model.load_state_dict(dict_pkl)
        self._model.eval()

    def load_ref_dataset(self, ref_dataset_path: str, weights_arr: list):
        img_files = self.get_all_image_files(ref_dataset_path)

        for isp_i, isp_s in zip(img_files, weights_arr):
            score = np.array(isp_s)
            samples = self.get_img_score(isp_i, score)

            if img_tensor is None:
                img_tensor = samples["img"].unsqueeze(0)
                gt_tensor = samples["gt"].type(torch.FloatTensor).unsqueeze(0)
            else:
                img_tensor = torch.cat((img_tensor, samples["img"].unsqueeze(0)), dim=0)
                gt_tensor = torch.cat(
                    (gt_tensor, samples["gt"].type(torch.FloatTensor).unsqueeze(0)),
                    dim=0,
                )

        img = img_tensor.squeeze(0).cuda()
        label = gt_tensor.squeeze(0).cuda()
        self._model.forward_prompt(img, label.reshape(-1, 1), "example")

    @staticmethod
    def get_all_image_files(folder_path: str) -> list[Path]:
        """"""
        folder = Path(folder_path)
        files_pngs = list(folder.rglob("*.png"))
        files_jpgs = list(folder.rglob("*.jpg"))
        files = files_pngs + files_jpgs

        test_match = re.search(r"(\d+)", str(files[0].stem))
        digit_checker = int(test_match.group(1)) if test_match else False

        if not digit_checker:
            sorted_files = sorted(files, key=lambda f: f.name)
        else:
            sorted_files = sorted(files, key=lambda f: int(re.findall(r"\d+", f.name)[0]))

        if len(sorted_files) == 0:
            raise RuntimeWarning(f"No files were found in <{folder_path}>. Nothing to process!")

        return sorted_files

    def preproc_input_data(self, image: torch.Tensor, target_score: float = 0.0):
        pass

    def compute_quality(self):
        pass

