import os
from pathlib import Path
from huggingface_hub import snapshot_download


if __name__ == '__main__':
    checkpoint_dir_default = Path(os.path.join(os.curdir, "./PromptIQA/checkpoints"))
    if not checkpoint_dir_default.exists():
        checkpoint_dir_default.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id='Zevin2023/PromptIQA', local_dir=checkpoint_dir_default)
