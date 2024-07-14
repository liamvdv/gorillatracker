import copy
from typing import Callable

from gorillatracker.model.base_module import BaseModule
import torch
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
import timm
from torch import nn
import wandb
from PIL import Image

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM

from gorillatracker.data.utils import flatten_batch
from gorillatracker import type_helper as gtypes
from gorillatracker.metrics import get_n_samples_from_dataloader, tensor_to_image
from gorillatracker.transform_utils import PlanckianJitter
from gorillatracker.utils.l2sp_regularisation import L2_SP, L2


# TODO(rob2u): add type hints
# TODO(rob2u): use actual MAE model parameters
# TODO(rob2u): check if normalized reconstruction targets are used
class MaskedVisionTransformer(BaseModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        decoder_dim = 512 # decoder_width, default
        decoder_depth = 8 # default
        decoder_num_heads = 16 # default
        mlp_ratio = 4.0 # default
        proj_drop_rate = 0.0 # default
        attn_drop_rate = 0.0 # default
        mask_ratio = 0.75 # default
        
        vit = timm.create_model(
            "timm/vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )
        
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.backbone.vit = timm.create_model(
            "timm/vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        ) # NOTE(rob2u): workaround to keep the pretrained weights
        
        self.patch_size = vit.patch_embed.patch_size[0]
        self.mask_ratio = mask_ratio
        self.sequence_length = self.backbone.sequence_length
        
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        print(self.loss_mode)
        self.l2sp = False
        loss_mode = self.loss_mode
        if "/l2sp" in loss_mode:
            self.l2sp = True
            loss_mode = loss_mode.replace("/l2sp", "")
                    
        if loss_mode == "mae_mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Loss mode {self.loss_mode} not supported")
            
        model_cpy = copy.deepcopy(self.backbone.vit)
        model_cpy.head = nn.Identity()
        
        self.l2sp_backbone = L2_SP(
            model_cpy,
            kwargs["path_to_pretrained_weights"],
            kwargs["l2_alpha"],
            kwargs["l2_beta"],
        )
        
        self.l2_decoder = L2(
            self.decoder,
            kwargs["l2_alpha"], 
        )
            
    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        
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

    def training_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        _, flat_images, _ = flatten_batch(batch)
        
        batch_size = flat_images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=flat_images.device,
        )
        
        flat_images, idx_keep, idx_mask = flat_images.to(self.accelerator), idx_keep.to(self.accelerator), idx_mask.to(self.accelerator)
        
        x_encoded = self.forward_encoder(images=flat_images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask,
        )
        
        # get image patches for masked tokens
        patches = utils.patchify(flat_images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1).to(self.accelerator)
        
        reg_term_decoder = self.l2_decoder(self.decoder) if self.l2sp else 0.0
        reg_term_encoder = self.l2sp_backbone(self.backbone.vit) if self.l2sp else 0.0
        mse_loss = self.criterion(x_pred, target)
        total_loss = mse_loss + reg_term_decoder + reg_term_encoder
        
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
            
            original_img, idx_keep, idx_mask = original_img.to(self.accelerator), idx_keep.to(self.accelerator), idx_mask.to(self.accelerator)
            x_encoded = self.forward_encoder(images=original_img, idx_keep=idx_keep)
            x_pred = self.forward_decoder(
                x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
            )
            
            # get image patches for masked tokens
            patches = utils.patchify(original_img, self.patch_size)
            patches_masked = utils.set_at_index(patches, idx_mask - 1, 0) # exclude class token
            masked_img = utils.unpatchify(patches_masked, self.patch_size)
            
            # get image patches for predicted tokens
            patches_pred = utils.set_at_index(patches, idx_mask - 1, x_pred)
            reconstructed_img = utils.unpatchify(patches_pred, self.patch_size)
            
            # unnormalize images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original_img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original_img.device)
            original_img = original_img * std + mean
            masked_img = masked_img * std + mean
            reconstructed_img = reconstructed_img * std + mean
            
            img_pil = tensor_to_image(original_img[0])
            masked_img_pil = tensor_to_image(masked_img[0])
            reconstructed_img_pil = tensor_to_image(reconstructed_img[0])
            
            artifacts = [
                wandb.Image(img_pil, caption=f"original_{i}"),
                wandb.Image(masked_img_pil, caption=f"masked_original_{i}"), 
                wandb.Image(reconstructed_img_pil, caption=f"reconstruction_{i}")
            ]
            self.wandb_run.log({f"epoch_{self.trainer.current_epoch}_nlet_{1+i}": artifacts})
    
    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]: # TODO
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

if __name__ == "__main__":
    wandb_run = None
    data_module = None
    mae = MaskedVisionTransformer(
        wandb_run=wandb_run,
        data_module=data_module,
        loss_mode="mse",
    )
    # test = (torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224))
    sample_img = Image.open("/workspaces/gorillatracker/data/supervised/cxl_all/face_images_square/AP00_R066_20221118_110aSilver.png")
    sample_img = transforms_v2.Resize((224, 224))(sample_img)
    sample_img = transforms_v2.ToTensor()(sample_img).unsqueeze(0)
    test = (sample_img, sample_img)
    test_label = (torch.tensor([0]), torch.tensor([1]))
    loss = mae.training_step((("0", "1"), test, test_label), 0)
    print(loss)
    