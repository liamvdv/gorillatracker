import time
from itertools import chain
from typing import Any, Callable, Optional, Union

import timm
import torch
import wandb
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from torch import nn
from torchvision.transforms import v2 as transforms_v2

from gorillatracker import type_helper as gtypes
from gorillatracker.data.utils import flatten_batch
from gorillatracker.losses.arcface_loss import ArcFaceLoss
from gorillatracker.metrics import get_n_samples_from_dataloader, tensor_to_image
from gorillatracker.model.base_module import BaseModule
from gorillatracker.utils.l2sp_regularisation import L2, L2_SP

BatchDatasetidxType = tuple[tuple[tuple[str], ...], tuple[torch.Tensor], tuple[torch.Tensor], tuple[tuple[int], ...]]


def flatten_batch_datasetidx(
    batch: BatchDatasetidxType,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor, tuple[int, ...]]:
    ids, images, labels, dataset_idxs = batch
    # transform ((a1, a2), (p1, p2), (n1, n2)) to (a1, a2, p1, p2, n1, n2)
    flat_ids: tuple[str, ...] = tuple(chain.from_iterable(ids))
    flat_dsidxs: tuple[int, ...] = tuple(chain.from_iterable(dataset_idxs))
    # transform ((a1, a2), (p1, p2), (n1, n2)) to (a1, a2, p1, p2, n1, n2)
    flat_labels = torch.cat(labels)
    # transform ((a1: Tensor, a2: Tensor), (p1: Tensor, p2: Tensor), (n1: Tensor, n2: Tensor))  to (a1, a2, p1, p2, n1, n2)
    flat_images = torch.cat(images)
    return flat_ids, flat_images, flat_labels, flat_dsidxs


class MaskedVisionTransformer(BaseModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.val_img: Union[None, torch.Tensor] = None
        self.use_positives = True

        decoder_dim = 512  # decoder_width, default
        decoder_depth = 4  # default
        decoder_num_heads = 16  # default
        mlp_ratio = 4.0  # default
        proj_drop_rate = 0.0  # default
        attn_drop_rate = 0.0  # default
        mask_ratio = 0.75  # default: 0.75
        vit = timm.create_model(
            "timm/vit_large_patch16_224.mae",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.backbone.vit = timm.create_model(
            "timm/vit_large_patch16_224.mae",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )  # NOTE(rob2u): workaround to keep the pretrained weights
        # # freeze backbone
        # for param in self.backbone.vit.parameters():
        #     param.requires_grad = False

        self.patch_size = vit.patch_embed.patch_size[0]
        self.mask_ratio = mask_ratio
        self.sequence_length = self.backbone.sequence_length

        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,  # usually vit.embed_dim TODO(rob2u) add linear projection to make shapes fit
            decoder_embed_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        if self.use_positives:
            self.patch_projection = nn.Linear(self.patch_size**2 * 3, vit.embed_dim)

        print(self.loss_mode)
        self.l2sp = False
        loss_mode = self.loss_mode
        if "/l2sp" in loss_mode:
            self.l2sp = True
            loss_mode = loss_mode.replace("/l2sp", "")

        if loss_mode.startswith("mae_mse"):
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Loss mode {self.loss_mode} not supported")

        self.backbone.vit.head = nn.Identity()
        self.l2sp_backbone: Union[L2_SP, Callable[[Any], float]]
        self.l2_decoder: Union[L2, L2_SP, Callable[[Any], float]]
        if self.l2sp:
            self.l2sp_backbone = L2_SP(
                self.backbone.vit,
                kwargs["path_to_pretrained_weights"],
                # kwargs["l2_alpha"],
                0.0,
                kwargs["l2_beta"],
            )

            self.l2_decoder = L2(
                self.decoder,
                kwargs["l2_alpha"],
            )
        else:
            self.l2sp_backbone = lambda x: 0.0
            self.l2_decoder = lambda x: 0.0

        self.mse_factor = 100.0

        self.supervised_loss_factor = 30.0

        if "/arcface" in self.loss_mode:
            self.supervised_loss = ArcFaceLoss(
                embedding_size=kwargs["embedding_size"],
                num_classes=kwargs["num_classes"][0],
                class_distribution=kwargs["class_distribution"],
                angle_margin=kwargs["margin"],
                additive_margin=0.0,
                s=kwargs["s"],
                accelerator=self.accelerator,  # type: ignore
                k_subcenters=kwargs["k_subcenters"],
                use_focal_loss=kwargs["use_focal_loss"],
                label_smoothing=kwargs["label_smoothing"],
                use_class_weights=kwargs["use_class_weights"],
                use_dist_term=kwargs["use_dist_term"],
                purpose="train",
            )

    def forward_encoder(self, images: torch.Tensor, idx_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded: torch.Tensor, idx_keep: torch.Tensor, idx_mask: torch.Tensor) -> torch.Tensor:
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))

        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone.vit.forward_features(images)
        x = self.backbone.vit.forward_head(x, pre_logits=True)
        return x

    def training_step(self, batch: Union[gtypes.NletBatch, BatchDatasetidxType], batch_idx: int) -> torch.Tensor:  # type: ignore
        supervised_loss = torch.tensor(0.0).to(self.accelerator)
        if "/arcface" in self.loss_mode:
            _, flat_images, flat_labels, dataset_idxs = flatten_batch_datasetidx(batch)  # type: ignore
            flat_images_mae = flat_images[[dataset_idx == 0 for dataset_idx in dataset_idxs]]
            flat_images_supervised = flat_images[[dataset_idx == 1 for dataset_idx in dataset_idxs]]
            flat_labels_supervised = flat_labels[[dataset_idx == 1 for dataset_idx in dataset_idxs]]
            if flat_images_supervised.shape[0] > 0:
                embeddings = self.forward(images=flat_images_supervised)
                supervised_loss = self.supervised_loss(embeddings=embeddings, labels=flat_labels_supervised)[0]
        else:
            _, flat_images_mae, _ = flatten_batch(batch)  # type: ignore

        if self.use_positives:
            flat_images_mae_positive = flat_images_mae[len(flat_images_mae) // 2 :]
            flat_images_mae = flat_images_mae[: len(flat_images_mae) // 2]

        batch_size = flat_images_mae.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=flat_images_mae.device,
        )

        flat_images_mae, idx_keep, idx_mask = (
            flat_images_mae.to(self.accelerator),
            idx_keep.to(self.accelerator),
            idx_mask.to(self.accelerator),
        )

        x_encoded = self.forward_encoder(images=flat_images_mae, idx_keep=idx_keep)

        if self.use_positives:
            cls_token = x_encoded[:, 0]
            patches_pos = utils.patchify(flat_images_mae_positive, self.patch_size)
            batch_size, num_patches, patch_size = patches_pos.shape

            patches_pos = patches_pos.view(-1, patch_size)
            patches_pos = self.patch_projection(patches_pos)
            patches_pos = patches_pos.view(batch_size, num_patches, -1)

            seq_pos = torch.cat([cls_token.unsqueeze(1), patches_pos], dim=1)
            x_encoded = utils.get_at_index(seq_pos, idx_keep).to(self.accelerator)

        # get image patches for masked tokens
        patches = utils.patchify(flat_images_mae, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1).to(self.accelerator)

        x_pred = self.forward_decoder(
            x_encoded=x_encoded,
            idx_keep=idx_keep,
            idx_mask=idx_mask,
        )

        reg_term_decoder = self.l2_decoder(self.decoder) if self.l2sp else 0.0
        reg_term_encoder = self.l2sp_backbone(self.backbone.vit) if self.l2sp else 0.0
        mse_loss = self.criterion(x_pred, target)
        total_loss = (
            mse_loss * self.mse_factor
            + reg_term_decoder
            + reg_term_encoder
            + supervised_loss * self.supervised_loss_factor
        )

        self.log("train/supervised_loss", supervised_loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/mse_loss", mse_loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/reg_term_decoder", reg_term_decoder, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/reg_term_encoder", reg_term_encoder, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True, sync_dist=True)
        return total_loss

    def on_train_epoch_start(self) -> None:
        samples = get_n_samples_from_dataloader(self.dm.train_dataloader(), n_samples=2)

        for i, sample in enumerate(samples):
            original_img = sample[1][0].unsqueeze(0)

            idx_keep, idx_mask = utils.random_token_mask(
                size=(1, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=original_img.device,
            )

            original_img, idx_keep, idx_mask = (
                original_img.to(self.accelerator),
                idx_keep.to(self.accelerator),
                idx_mask.to(self.accelerator),
            )
            x_encoded = self.forward_encoder(images=original_img, idx_keep=idx_keep)

            if self.use_positives:
                cls_token = x_encoded[:, 0]
                patches_pos = utils.patchify(original_img, self.patch_size)
                batch_size, num_patches, patch_size = patches_pos.shape
                patches_pos = patches_pos.view(-1, patch_size)

                patches_pos = self.patch_projection(patches_pos)
                patches_pos = patches_pos.view(batch_size, num_patches, -1)

                patches_pos = torch.cat([cls_token.unsqueeze(1), patches_pos], dim=1)
                x_encoded = utils.get_at_index(patches_pos, idx_keep).to(self.accelerator)

            x_pred = self.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)

            # get image patches for masked tokens
            patches = utils.patchify(original_img, self.patch_size)
            patches_masked = utils.set_at_index(patches, idx_mask - 1, 0)  # exclude class token
            masked_img = utils.unpatchify(patches_masked, self.patch_size)

            # get image patches for predicted tokens
            patches_pred = utils.set_at_index(patches, idx_mask - 1, x_pred)
            reconstructed_img = utils.unpatchify(patches_pred, self.patch_size)

            # unnormalize images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)

            original_img = original_img * std + mean
            original_img = original_img.clamp(0, 1)
            masked_img = masked_img * std + mean
            masked_img = masked_img.clamp(0, 1)
            reconstructed_img = reconstructed_img * std + mean
            reconstructed_img = reconstructed_img.clamp(0, 1)

            img_pil = tensor_to_image(original_img[0])
            masked_img_pil = tensor_to_image(masked_img[0])
            reconstructed_img_pil = tensor_to_image(reconstructed_img[0])

            artifacts = [
                wandb.Image(img_pil, caption=f"original_{i}"),
                wandb.Image(masked_img_pil, caption=f"masked_original_{i}"),
                wandb.Image(reconstructed_img_pil, caption=f"reconstruction_{i}"),
            ]
            self.wandb_run.log({f"epoch_{self.trainer.current_epoch}_nlet_{1+i}": artifacts})

    def validation_step(
        self,
        batch: gtypes.NletBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:

        if self.use_positives:
            ids, images, labels = batch
            ids, images, labels = (
                (ids[0],),
                (images[0],),
                (labels[0],),
            )
            batch = (ids, images, labels)

        if self.val_img is None:
            self.val_img = batch[1][0][0].unsqueeze(0)
        output = super().validation_step(batch, batch_idx, dataloader_idx)

        if output != torch.tensor(0.0):
            return output
        else:
            _, flat_images_mae, _ = flatten_batch(batch)

            batch_size = flat_images_mae.shape[0]
            idx_keep, idx_mask = utils.random_token_mask(
                size=(batch_size, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=flat_images_mae.device,
            )

            flat_images_mae, idx_keep, idx_mask = (
                flat_images_mae.to(self.accelerator),
                idx_keep.to(self.accelerator),
                idx_mask.to(self.accelerator),
            )

            x_encoded = self.forward_encoder(images=flat_images_mae, idx_keep=idx_keep)

            x_pred = self.forward_decoder(
                x_encoded=x_encoded,
                idx_keep=idx_keep,
                idx_mask=idx_mask,
            )

            # get image patches for masked tokens
            patches = utils.patchify(flat_images_mae, self.patch_size)
            # must adjust idx_mask for missing class token
            target = utils.get_at_index(patches, idx_mask - 1).to(self.accelerator)

            reg_term_decoder = self.l2_decoder(self.decoder) if self.l2sp else 0.0
            reg_term_encoder = self.l2sp_backbone(self.backbone.vit) if self.l2sp else 0.0
            mse_loss = self.criterion(x_pred, target)
            total_loss = mse_loss * self.mse_factor + reg_term_decoder + reg_term_encoder

            self.log("val/loss", total_loss, on_epoch=True)

            return total_loss

    def on_validation_epoch_end(self, dataloader_idx: int = 0) -> None:
        if self.val_img is not None:
            original_img = self.val_img
            idx_keep, idx_mask = utils.random_token_mask(
                size=(1, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=original_img.device,
            )

            original_img, idx_keep, idx_mask = (
                original_img.to(self.accelerator),
                idx_keep.to(self.accelerator),
                idx_mask.to(self.accelerator),
            )
            x_encoded = self.forward_encoder(images=original_img, idx_keep=idx_keep)

            if self.use_positives:
                cls_token = x_encoded[:, 0]
                patches_pos = utils.patchify(original_img, self.patch_size)
                batch_size, num_patches, patch_size = patches_pos.shape
                patches_pos = patches_pos.view(-1, patch_size)

                patches_pos = self.patch_projection(patches_pos)
                patches_pos = patches_pos.view(batch_size, num_patches, -1)

                patches_pos = torch.cat([cls_token.unsqueeze(1), patches_pos], dim=1)
                x_encoded = utils.get_at_index(patches_pos, idx_keep).to(self.accelerator)

            x_pred = self.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)

            # get image patches for masked tokens
            patches = utils.patchify(original_img, self.patch_size)
            patches_masked = utils.set_at_index(patches, idx_mask - 1, 0)  # exclude class token
            masked_img = utils.unpatchify(patches_masked, self.patch_size)

            # get image patches for predicted tokens
            patches_pred = utils.set_at_index(patches, idx_mask - 1, x_pred)
            reconstructed_img = utils.unpatchify(patches_pred, self.patch_size)

            # unnormalize images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)

            original_img = original_img * std + mean
            original_img = original_img.clamp(0, 1)
            masked_img = masked_img * std + mean
            masked_img = masked_img.clamp(0, 1)
            reconstructed_img = reconstructed_img * std + mean
            reconstructed_img = reconstructed_img.clamp(0, 1)

            img_pil = tensor_to_image(original_img[0])
            masked_img_pil = tensor_to_image(masked_img[0])
            reconstructed_img_pil = tensor_to_image(reconstructed_img[0])

            artifacts = [
                wandb.Image(img_pil, caption="original"),
                wandb.Image(masked_img_pil, caption="masked_original"),
                wandb.Image(reconstructed_img_pil, caption="reconstruction"),
            ]
            self.wandb_run.log({f"val_epoch_{self.trainer.current_epoch}": artifacts})

        return super().on_validation_epoch_end(dataloader_idx)

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms_v2.Compose(
            [
                # transforms.ToPILImage(),
                # PlanckianJitter(),
                # transforms.ToTensor(),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
            ]
        )
