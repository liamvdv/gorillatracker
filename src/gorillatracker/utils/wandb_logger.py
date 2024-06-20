import dataclasses
import os
import re
import time
from typing import Any, Optional

from lightning.pytorch.loggers.wandb import WandbLogger
from print_on_steroids import logger

from gorillatracker.args import TrainingArgs


class WandbLoggingModule:
    def __init__(
        self,
        args: TrainingArgs,
        wandb_entity: str = "gorillas",
    ) -> None:
        self._args_assertions(args)
        self.args = args
        self.wandb_entity = wandb_entity
        self.run_name = args.run_name
        self.project_name = args.project_name

        if args.offline or args.fast_dev_run or args.data_preprocessing_only:
            os.environ["WANDB_MODE"] = "dryrun"

    def _args_assertions(self, args: TrainingArgs) -> None:
        asserations = [
            (args.run_name, "No run name specified with `--run_name`. Please specify a run name."),
            (len(args.run_name.split("-")) >= 2, "Run name must be of the form <Issue Number>-<Purpose>"),
            (
                args.project_name,
                "No project name specified with `--project_name`. Please specify a project name.",
            ),
            (
                len(args.project_name.split("-")) >= 4,
                "Project name must be of the form <Function>-<Backbone>-<Dataset>-<Set-Type>",
            ),
        ]

        for assertion in asserations:
            assert assertion[0], assertion[1]

    def construct_logger(self, log_code: bool = False) -> WandbLogger:
        """
        The `construct_logger` function creates a WandbLogger object for logging
        training progress and hyperparameters to the Weights & Biases platform.

        Args:
          log_code (bool): The `log_code` parameter is a boolean flag that determines
        whether to log the code used for training in the WandbLogger. If `log_code` is
        set to `True`, the code will be logged; otherwise, it will not be logged.
        Defaults to False

        Returns:
          an instance of the `WandbLogger` class.
        """
        wandb_extra_args = dict(name=self.run_name)

        if (
            self.args.saved_checkpoint_path
            and self.args.resume
            and self.check_checkpoint_path_for_wandb(self.args.saved_checkpoint_path)
        ):
            logger.info("Resuming training from W&B")
            id = self.check_checkpoint_path_for_wandb(self.args.saved_checkpoint_path)
            assert id is not None
            wandb_extra_args = dict(id=id, resume="must")  # resume W&B run

        wandb_logger = WandbLogger(
            project=self.project_name,
            entity=self.wandb_entity,
            log_model="all",
            tags=self.args.wandb_tags,
            save_dir="logs/",
            **wandb_extra_args,  # type: ignore
        )
        wandb_logger.log_hyperparams(dataclasses.asdict(self.args))

        logger.info(self.args)
        if self.args.resume and not self.args.offline:
            assert wandb_logger.version is not None, "W&B logger version is None. This should not happen."
            wandb_logger.experiment.name = self.args.run_name + "-" + time.strftime("%Y-%m-%d-%H-%M-%S")

        return wandb_logger

    def check_for_wandb_checkpoint_and_download_if_necessary(
        self,
        checkpoint_path: str,
        wandb_run_instance: Any,
        suffix: str = "/model.ckpt",
    ) -> str:
        """
        Checks the provided checkpoint_path for the wandb regex r\"wandb:.*\".
        If matched, download the W&B artifact indicated by the id in the provided string and return its path.
        If not, just returns provided string.

        Path format: wandb:model_id:tag
        """
        wandb_model_id_regex = r"wandb:.*"
        if re.search(wandb_model_id_regex, checkpoint_path):
            wandb_model_id = checkpoint_path.split(":")[1]
            model_tag = checkpoint_path.split(":")[2] if len(checkpoint_path.split(":")) == 3 else "latest"

            """
            Only the main process should download the artifact in DDP. We add this environment variable as a guard. 
            This works only if this function is called first on the main process.
            """
            if os.environ.get(f"MY_WANDB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"):
                checkpoint_path = os.environ[f"MY_WANDB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"]
            else:
                artifact = wandb_run_instance.use_artifact(
                    f"{self.wandb_entity}/{self.project_name}/model-{wandb_model_id}:{model_tag}"
                )
                checkpoint_path = artifact.download() + suffix
                logger.info(f"Path of downloaded W&B artifact: {checkpoint_path}")
                os.environ[f"MY_WANDB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"] = checkpoint_path
        return checkpoint_path

    def check_checkpoint_path_for_wandb(self, checkpoint_path: str) -> Optional[str]:
        wandb_model_id_regex = r"wandb:.*"
        if re.search(wandb_model_id_regex, checkpoint_path):
            wandb_model_id = checkpoint_path.split(":")[1]
            return wandb_model_id
        return None
