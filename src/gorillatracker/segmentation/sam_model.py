import lightning as L
import numpy as np
import torch
from print_on_steroids import logger
from segment_anything import sam_model_registry
from torch.nn.functional import normalize, threshold
from torch.optim import AdamW
from transformers.optimization import get_scheduler


class SAMDecoderFineTuner(L.LightningModule):
    def __init__(
        self,
        model_type: str,
        checkpoint: str,
        from_scratch: bool,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_period: int,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
    ) -> None:
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_schedule = lr_schedule
        self.warmup_period = warmup_period
        self.epsilon = epsilon

        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        # TODO Which loss does make sense here
        loss = torch.nn.BCELoss()
        self.model.loss = loss

    def forward(self, batch):
        image_embedding, input_size, original_image_size, sparse_embeddings, dense_embeddings, _ = batch
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = self.model.postprocess_masks(low_res_masks, input_size, original_image_size)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        return binary_mask

    def _convert_to_binary_mask(sel, ground_truth_mask):
        gt_mask_resized = torch.from_numpy(
            np.resize(ground_truth_mask, (1, 1, ground_truth_mask.shape[0], ground_truth_mask.shape[1]))
        )
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
        return gt_binary_mask

    def training_step(self, batch, batch_idx):
        binary_mask = self.forward(batch)
        ground_truth_mask = batch[-1]

        gt_binary_mask = self._convert_to_binary_mask(ground_truth_mask)
        loss = self.model.loss(binary_mask, gt_binary_mask)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        binary_mask = self.forward(batch)
        ground_truth_mask = batch[-1]

        gt_binary_mask = self._convert_to_binary_mask(ground_truth_mask)
        loss = self.model.loss(binary_mask, gt_binary_mask)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup steps: {self.warmup_period}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,  # You can also tune this
        )

        scheduler_name = self.lr_schedule
        if scheduler_name == "constant" and self.warmup_period > 0:
            scheduler_name += "_with_warmup"
        scheduler = get_scheduler(
            scheduler_name,
            optimizer,
            num_warmup_steps=int(self.warmup_period),
            num_training_steps=self.trainer.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
