import yaml
import re
from pathlib import Path
from PIL import Image
from typing import Any
from time import time

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from huggingface_hub import snapshot_download
from PromptIQA.models import promptiqa
from PromptIQA.utils.dataset.process import Normalize
from PromptIQA.utils.toolkit import *
import warnings

warnings.filterwarnings("ignore")


class PromptIQAPipeline:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._model: promptiqa.PromptIQA = None
        self._config_data: dict = None
        self._cur_path = Path(__file__).parent

    def _download_model(self):
        checkpoint_dir_default = self._cur_path / Path(self._config_data["checkpoint_path"]).parent
        if not checkpoint_dir_default.exists():
            checkpoint_dir_default.mkdir(parents=True, exist_ok=True)

        snapshot_download(repo_id='Zevin2023/PromptIQA', local_dir=checkpoint_dir_default)

    def _load_config(self) -> None:
        """ Function for loading the data from the .yaml configuration file. """
        config_path = self._cur_path / Path("prompt_iqa_conf.yml")
        with open(config_path, "r") as file:
            self._config_data = yaml.safe_load(file)
        assert self._config_data != {}

    def load_pipeline(self) -> None:
        logger.info("Loading prompt-IQA pipeline.")
        t1 = time()
        self._load_config()

        checkpoint_dir_default = self._cur_path / Path(self._config_data["checkpoint_path"])
        self.load_model(checkpoint_dir_default.as_posix())

        ref_dataset_path = self._cur_path / Path(self._config_data["reference_dataset_path"])
        self.load_ref_dataset(ref_dataset_path.as_posix(), self._config_data["reference_weights"])
        t2 = time()
        logger.info(f"It took: {t2 - t1} sec \n")

    def load_model(self, ckpt_path: str) -> None:
        self._download_model()
        self._model = promptiqa.PromptIQA()

        dict_pkl = {}
        for key, value in torch.load(ckpt_path, map_location="cpu")["state_dict"].items():
            dict_pkl[key[7:]] = value
        self._model.load_state_dict(dict_pkl)
        self._model.to(self._device)
        self._model.eval()

    def load_ref_dataset(self, ref_dataset_path: str, weights_arr: list) -> None:
        img_files = self.get_all_image_files(ref_dataset_path)

        loaded_torch_imgs = []
        loaded_torch_scores = []

        for isp_i, isp_s in zip(img_files, weights_arr):
            image = torch.tensor(np.asarray(Image.open(isp_i.as_posix())))
            score = torch.tensor(isp_s)

            samples = self.preproc_input_data(image, score)
            loaded_torch_imgs.append(samples["img"].unsqueeze(0))
            loaded_torch_scores.append(samples["gt"].type(torch.FloatTensor).unsqueeze(0))

        img_tensor = torch.cat(loaded_torch_imgs, dim=0)
        gt_tensor = torch.cat(loaded_torch_scores, dim=0)

        input_img_tensor = img_tensor.squeeze(0).to(self._device)
        labels_tensor = gt_tensor.squeeze(0).to(self._device)

        self._model.forward_prompt(input_img_tensor, labels_tensor.reshape(-1, 1), "example")

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

    @staticmethod
    def preproc_input_data(image: torch.Tensor, target_score: torch.Tensor = 0.0, img_size: int = 224) -> dict:
        proc_image = (image / 255.0).unsqueeze(0)
        proc_image = proc_image.permute(0, 3, 1, 2).to(torch.float32)
        proc_image = F.interpolate(proc_image, size=(img_size, img_size), mode="bicubic", align_corners=False)

        samples = {"img": proc_image.squeeze(0), "gt": target_score}
        normalize_func = Normalize(0.5, 0.5)
        samples = normalize_func(samples)
        return samples

    def compute_quality(self, image: torch.Tensor) -> Any:
        samples = self.preproc_input_data(image)
        proc_image = samples["img"].unsqueeze(0).to(self._device)
        predicted_score = self._model.inference(proc_image, "example")
        return predicted_score
