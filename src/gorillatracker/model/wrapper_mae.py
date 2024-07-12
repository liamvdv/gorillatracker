from typing import Callable

from gorillatracker.model.base_module import BaseModule
import torch
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

# TODO(rob2u): adapt to BaseModule interface
# TODO(rob2u): add type hints
# TODO(rob2u): use actual MAE model parameters
# TODO(rob2u): check if normalized reconstruction targets are used
class MAE(BaseModule):
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
        self.patch_size = vit.patch_embed.patch_size[0]
        self.mask_ratio = mask_ratio
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.backbone.vit = vit # NOTE(rob2u): workaround to keep the pretrained weights
        self.sequence_length = self.backbone.sequence_length # NOTE(rob2u): exlude class token
        
        print(f"sequence_length: {self.sequence_length}")
        print(f"num_patches: {vit.patch_embed.num_patches}")
        
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches, # NOTE(rob2u): exlude class token
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.criterion = nn.MSELoss()

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
        return self.backbone(images)

    def training_step(self, batch: gtypes.NletBatch, batch_idx: int) -> torch.Tensor:
        _, flat_images, _ = flatten_batch(batch)
        
        batch_size = flat_images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=flat_images.device,
        )
        
        x_encoded = self.forward_encoder(images=flat_images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask,
        )

        # get image patches for masked tokens
        patches = utils.patchify(flat_images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        
        loss = self.criterion(x_pred, target)
        return loss, -1, -1

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim
    
    def on_train_epoch_start(self, sample_img) -> None: # TODO
        # samples = get_n_samples_from_dataloader(self.dm.train_dataloader(), n_samples=1)
        samples = [sample_img]
        
        for i, img in enumerate(samples):
            # img = sample[1][0]
            original = img
            idx_keep, idx_mask = utils.random_token_mask(
                size=(1, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=img.device,
            )
            print(idx_keep.shape)
            print(idx_mask.shape)
            
            x_encoded = self.forward_encoder(images=img, idx_keep=idx_keep)
            x_pred = self.forward_decoder(
                x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
            )
            
            # get image patches for masked tokens
            patches = utils.patchify(img, self.patch_size)
            patches_masked = utils.set_at_index(patches, idx_mask - 1, 0) # exclude class token
            masked_img = utils.unpatchify(patches_masked, self.patch_size)
            
            # get image patches for predicted tokens
            patches_pred = utils.set_at_index(patches, idx_mask - 1, x_pred)
            reconstructed_img = utils.unpatchify(patches_pred, self.patch_size)
            
            img_pil = tensor_to_image(original[0])
            masked_img_pil = tensor_to_image(masked_img[0])
            reconstructed_img_pil = tensor_to_image(reconstructed_img[0])
            
            img_pil.save(f"original_{i}.png")
            masked_img_pil.save(f"masked_original_{i}.png")
            reconstructed_img_pil.save(f"reconstruction_{i}.png")
            
            # artifacts = [
            #     wandb.Image(img, caption=f"original_{i}"),
            #     wandb.Image(masked_img, caption=f"masked_original_{i}"), 
            #     wandb.Image(reconstruction, caption=f"reconstruction_{i}")
            # ]
            # self.wandb_run.log({f"epoch_{self.trainer.current_epoch}_nlet_{1+i}": artifacts})
    
    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]: # TODO
        return transforms_v2.Compose(
            [
                PlanckianJitter(),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(192, scale=(0.75, 1.0)),
            ]
        )

if __name__ == "__main__":
    wandb_run = None
    data_module = None
    mae = MAE(
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
    mae.on_train_epoch_start(sample_img)
    # size = (32, 257)
    # mask_ratio = 0.75
    # print(size)
    # print(mask_ratio)
    # idx_keep, idx_mask = utils.random_token_mask(
    #         size=size,
    #         mask_ratio=mask_ratio,
    #         mask_class_token=False,
    #     )
    # print(idx_keep.shape)
    # print(idx_mask.shape)
    