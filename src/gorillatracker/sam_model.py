import lightning as L
import torch
from print_on_steroids import logger
from segment_anything import sam_model_registry
from torch.nn.functional import normalize, threshold
from torch.optim import AdamW


class SamDecoderFineTuner(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        from_scratch: bool,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        lr_schedule: str,
        warmup_epochs: int,
        lr_decay: float,
        lr_decay_interval: int,
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
        self.warmup_epochs = warmup_epochs
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval
        self.epsilon = epsilon

        # infer model_type from model_name_or_path
        model_type = "_".join(model_name_or_path.split("/")[-1].split("_")[1:3])
        self.sam_model = sam_model_registry[model_type](checkpoint=model_name_or_path)
        # TODO Which loss does make sense here
        loss = torch.nn.BCELoss()
        self.sam_model.loss = loss

    def forward(self, batch):
        image_embeddings, input_sizes, original_image_sizes, sparse_embeddings, dense_embeddings, _ = batch

        binary_masks = []
        for i, (image_embedding, sparse_embedding, dense_embedding) in enumerate(
            zip(image_embeddings, sparse_embeddings, dense_embeddings)
        ):
            input_size = [t[i] for t in input_sizes]
            original_image_size = [t[i] for t in original_image_sizes]

            binary_mask = self._process_image(
                image_embedding, input_size, original_image_size, sparse_embedding, dense_embedding
            )
            binary_masks.append(binary_mask)

        return torch.stack(binary_masks)

    def _process_image(self, image_embedding, input_size, original_image_size, sparse_embedding, dense_embedding):
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
        )
        upscaled_masks = self.sam_model.postprocess_masks(low_res_masks, input_size, original_image_size)
        return normalize(threshold(upscaled_masks, 0.0, 0))

    def _convert_to_binary_mask(self, ground_truth_mask):
        # ground_truth_mask is expected to be of shape [batch_size, height, width]
        gt_mask_resized = ground_truth_mask.view(
            ground_truth_mask.shape[0], 1, ground_truth_mask.shape[1], ground_truth_mask.shape[2]
        )
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
        return gt_binary_mask

    def _calculate_batch_loss(self, batch):
        binary_mask = self.forward(batch)
        binary_mask = binary_mask.squeeze(2)
        ground_truth_mask = batch[-1]
        gt_binary_mask = self._convert_to_binary_mask(ground_truth_mask)
        loss = self.sam_model.loss(binary_mask, gt_binary_mask)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_batch_loss(batch)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_batch_loss(batch)
        self.log("val/loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup epochs: {self.warmup_epochs}"
            )

        named_parameters = list(self.sam_model.named_parameters())

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

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs * self.learning_rate
            else:
                num_decay_cycles = (epoch - self.warmup_epochs) // self.lr_decay_interval
                return (self.lr_decay**num_decay_cycles) * self.learning_rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": self.lr_decay_interval},
        }
