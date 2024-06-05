from typing import Any, Callable, Union

import torch

import gorillatracker.type_helper as gtypes

from .arcface_loss import AdaFaceLoss, ArcFaceLoss, ElasticArcFaceLoss, VariationalPrototypeLearning
from .dist_term_loss import CombinedLoss
from .l2sp import L2SPRegularization_Wrapper
from .offline_distillation_loss import OfflineResponseBasedLoss
from .triplet_loss import TripletLossOffline, TripletLossOfflineNative, TripletLossOnline


def get_loss(
    loss_mode: str,
    log_func: Callable[[str, float], None] = lambda x, y: None,
    **kw_args: Any,
) -> Callable[[torch.Tensor, torch.Tensor, gtypes.NletBatchValues], gtypes.LossPosNegDist]:
    l2sp = False
    if "l2sp" in loss_mode:
        loss_mode = loss_mode.replace("/l2sp", "")
        l2sp = True

    loss_module: Union[torch.nn.Module, None] = None

    assert not kw_args.get("use_dist_term", False) or "softmax" in loss_mode, "distance term only for softmax loss"
    assert kw_args["num_classes"] != -1 or "softmax" not in loss_mode, "num_classes must be set for softmax loss"
    assert (
        len(kw_args["class_distribution"]) > 0 or not kw_args["use_class_weights"]
    ), "class_distribution must be set for class weights"

    if loss_mode == "online/hard":
        loss_module = TripletLossOnline(mode="hard", margin=kw_args["margin"])
    elif loss_mode == "online/semi-hard":
        loss_module = TripletLossOnline(mode="semi-hard", margin=kw_args["margin"])
    elif loss_mode == "online/soft":
        loss_module = TripletLossOnline(mode="soft", margin=kw_args["margin"])
    elif loss_mode == "offline":
        loss_module = TripletLossOffline(margin=kw_args["margin"])
    elif loss_mode == "offline/native":
        loss_module = TripletLossOfflineNative(margin=kw_args["margin"])
    elif loss_mode == "distillation/offline/response-based":
        loss_module = OfflineResponseBasedLoss(teacher_model_wandb_link=kw_args["teacher_model_wandb_link"])
    elif loss_mode == "softmax/arcface":
        loss_module = ArcFaceLoss(
            embedding_size=kw_args["embedding_size"],
            angle_margin=kw_args["margin"],
            num_classes=kw_args["num_classes"],
            class_distribution=kw_args["class_distribution"],
            s=kw_args["s"],
            accelerator=kw_args["accelerator"],
            k_subcenters=kw_args["k_subcenters"],
            use_focal_loss=kw_args["use_focal_loss"],
            label_smoothing=kw_args["label_smoothing"],
            use_class_weights=kw_args["use_class_weights"],
        )
    elif loss_mode == "softmax/adaface":  # TODO
        loss_module = AdaFaceLoss(
            embedding_size=kw_args["embedding_size"],
            angle_margin=kw_args["margin"],
            num_classes=kw_args["num_classes"],
            class_distribution=kw_args["class_distribution"],
            s=kw_args["s"],
            accelerator=kw_args["accelerator"],
            k_subcenters=kw_args["k_subcenters"],
            use_focal_loss=kw_args["use_focal_loss"],
            label_smoothing=kw_args["label_smoothing"],
            use_class_weights=kw_args["use_class_weights"],
        )

    elif loss_mode == "softmax/elasticface":  # TODO
        loss_module = ElasticArcFaceLoss(
            embedding_size=kw_args["embedding_size"],
            angle_margin=kw_args["margin"],
            num_classes=kw_args["num_classes"],
            class_distribution=kw_args["class_distribution"],
            s=kw_args["s"],
            margin_sigma=0.078,
            accelerator=kw_args["accelerator"],
            k_subcenters=kw_args["k_subcenters"],
            use_focal_loss=kw_args["use_focal_loss"],
            label_smoothing=kw_args["label_smoothing"],
            use_class_weights=kw_args["use_class_weights"],
        )
    elif loss_mode == "softmax/vpl":
        loss_module = VariationalPrototypeLearning(
            embedding_size=kw_args["embedding_size"],
            num_classes=kw_args["num_classes"],
            class_distribution=kw_args["class_distribution"],
            batch_size=kw_args["batch_size"],
            s=kw_args["s"],
            margin=kw_args["margin"],
            delta_t=kw_args["delta_t"],
            mem_bank_start_epoch=kw_args["mem_bank_start_epoch"],
            accelerator=kw_args["accelerator"],
        )
    else:
        raise ValueError(f"Loss mode {loss_mode} not supported")

    if kw_args.get("use_dist_term", False):
        loss_module = CombinedLoss(
            arcface_loss=loss_module,  # type: ignore
            triplet_loss=TripletLossOnline(mode="soft", margin=1.0),
            lambda_=10.0,
            log_func=log_func,
        )

    if l2sp:
        return L2SPRegularization_Wrapper(
            loss=loss_module,
            model=kw_args["model"],
            path_to_pretrained_weights=kw_args["path_to_pretrained_weights"],
            alpha=kw_args["l2_alpha"],
            beta=kw_args["l2_beta"],
            log_func=log_func,
        )

    return loss_module
