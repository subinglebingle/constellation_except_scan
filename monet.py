from typing import Tuple

import torch
import torch.distributions as dists
from omegaconf import DictConfig
from torch import Tensor, nn

from abc import abstractmethod
from dataclasses import dataclass

from torch import nn

from dataclasses import dataclass
from typing import List, Optional, Union

from torch.nn import functional as F

def get_activation_module(activation_name: str, try_inplace: bool = True) -> nn.Module:
    if activation_name == "leakyrelu":
        act = torch.nn.LeakyReLU()
    elif activation_name == "elu":
        act = torch.nn.ELU()
    elif activation_name == "relu":
        act = torch.nn.ReLU(inplace=try_inplace)
    elif activation_name == "glu":
        act = torch.nn.GLU(dim=1)  # channel dimension in images
    elif activation_name == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activation_name == "tanh":
        act = torch.nn.Tanh()
    else:
        raise ValueError(f"Unknown activation name '{activation_name}'")
    return act

def make_sequential_from_config(
    input_channels: int,
    channels: List[int],
    kernels: Union[int, List[int]],
    batchnorms: Union[bool, List[bool]],
    bn_affines: Union[bool, List[bool]],
    paddings: Union[int, List[int]],
    strides: Union[int, List[int]],
    activations: Union[str, List[str]],
    output_paddings: Union[int, List[int]] = 0,
    conv_transposes: Union[bool, List[bool]] = False,
    return_params: bool = False,
    try_inplace_activation: bool = True,
) -> Union[nn.Sequential, Tuple[nn.Sequential, dict]]:
    # Make copy of locals and expand scalars to lists
    params = {k: v for k, v in locals().items()}
    params = _scalars_to_list(params)

    # Make sequential with the following order:
    # - Conv or conv transpose
    # - Optional batchnorm (optionally affine)
    # - Optional activation
    layers = []
    layer_infos = zip(
        params["channels"],
        params["batchnorms"],
        params["bn_affines"],
        params["kernels"],
        params["strides"],
        params["paddings"],
        params["activations"],
        params["conv_transposes"],
        params["output_paddings"],
    )
    for (
        channel,
        bn,
        bn_affine,
        kernel,
        stride,
        padding,
        activation,
        conv_transpose,
        o_padding,
    ) in layer_infos:
        if conv_transpose:
            layers.append(
                nn.ConvTranspose2d(
                    input_channels, channel, kernel, stride, padding, o_padding
                )
            )
        else:
            layers.append(nn.Conv2d(input_channels, channel, kernel, stride, padding))

        if bn:
            layers.append(nn.BatchNorm2d(channel, affine=bn_affine))
        if activation is not None:
            layers.append(
                get_activation_module(activation, try_inplace=try_inplace_activation)
            )

        # Input for next layer has half the channels of the current layer if using GLU.
        input_channels = channel
        if activation == "glu":
            input_channels //= 2

    if return_params:
        return nn.Sequential(*layers), params
    else:
        return nn.Sequential(*layers)
    
class INConvBlock(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        stride: int = 1,
        instance_norm: bool = True,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, 1, bias=not instance_norm)
        if instance_norm:
            self.instance_norm = nn.InstanceNorm2d(nout, affine=True)
        else:
            self.instance_norm = None
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        return self.act(x)

class UNet(nn.Module):
    def __init__(self, input_channels: int, num_blocks: int, filter_start: int = 32):
        super().__init__()
        c = filter_start
        if num_blocks == 4:
            self.down = nn.ModuleList(
                [
                    INConvBlock(input_channels + 1, c),
                    INConvBlock(c, 2 * c),
                    INConvBlock(2 * c, 2 * c),
                    INConvBlock(2 * c, 2 * c),  # no downsampling
                ]
            )
            self.up = nn.ModuleList(
                [
                    INConvBlock(4 * c, 2 * c),
                    INConvBlock(4 * c, 2 * c),
                    INConvBlock(4 * c, c),
                    INConvBlock(2 * c, c),
                ]
            )
        elif num_blocks == 5:
            self.down = nn.ModuleList(
                [
                    INConvBlock(4, c),
                    INConvBlock(c, c),
                    INConvBlock(c, 2 * c),
                    INConvBlock(2 * c, 2 * c),
                    INConvBlock(2 * c, 2 * c),  # no downsampling
                ]
            )
            self.up = nn.ModuleList(
                [
                    INConvBlock(4 * c, 2 * c),
                    INConvBlock(4 * c, 2 * c),
                    INConvBlock(4 * c, c),
                    INConvBlock(2 * c, c),
                    INConvBlock(2 * c, c),
                ]
            )
        elif num_blocks == 6:
            self.down = nn.ModuleList(
                [
                    INConvBlock(4, c),
                    INConvBlock(c, c),
                    INConvBlock(c, c),
                    INConvBlock(c, 2 * c),
                    INConvBlock(2 * c, 2 * c),
                    INConvBlock(2 * c, 2 * c),  # no downsampling
                ]
            )
            self.up = nn.ModuleList(
                [
                    INConvBlock(4 * c, 2 * c),
                    INConvBlock(4 * c, 2 * c),
                    INConvBlock(4 * c, c),
                    INConvBlock(2 * c, c),
                    INConvBlock(2 * c, c),
                    INConvBlock(2 * c, c),
                ]
            )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 2 * c, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4 * 4 * 2 * c),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(c, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(
                    act, scale_factor=0.5, mode="nearest", recompute_scale_factor=True
                )
            x_down.append(act)
        x_up = self.mlp(x_down[-1]).view(batch_size, -1, 4, 4)
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode="nearest")
        return self.final_conv(x_up)

@dataclass(eq=False, repr=False)
class EncoderNet(nn.Module):
    width: int
    height: int

    input_channels: int
    activations: str
    channels: List[int]
    batchnorms: List[bool]
    bn_affines: List[bool]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    mlp_hidden_size: int
    mlp_output_size: int

    def __post_init__(self):
        super().__init__()
        self.convs, params = make_sequential_from_config(
            self.input_channels,
            self.channels,
            self.kernels,
            self.batchnorms,
            self.bn_affines,
            self.paddings,
            self.strides,
            self.activations,
            return_params=True,
        )
        width = self.width
        height = self.height
        for kernel, stride, padding in zip(
            params["kernels"], params["strides"], params["paddings"]
        ):
            width = (width + 2 * padding - kernel) // stride + 1
            height = (height + 2 * padding - kernel) // stride + 1

        self.mlp = nn.Sequential(
            nn.Linear(self.channels[-1] * width * height, self.mlp_hidden_size),
            get_activation_module(self.activations, try_inplace=True),
            nn.Linear(self.mlp_hidden_size, self.mlp_output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x).flatten(1)
        x = self.mlp(x)
        return x

@dataclass(eq=False, repr=False)
class BroadcastDecoderNet(nn.Module):
    w_broadcast: int
    h_broadcast: int
    input_channels: int
    activations: List[Union[str, None]]
    channels: List[int]
    paddings: List[int]
    kernels: List[int]
    batchnorms: List[bool]
    bn_affines: List[bool]
    strides: Optional[List[int]] = None

    def __post_init__(self):
        super().__init__()
        self.parse_w_h()
        if self.strides is None:
            self.strides = 1  # type: int
        self.convs = make_sequential_from_config(
            self.input_channels,
            self.channels,
            self.kernels,
            self.batchnorms,
            self.bn_affines,
            self.paddings,
            self.strides,
            self.activations,
        )

        ys = torch.linspace(-1, 1, self.h_broadcast)
        xs = torch.linspace(-1, 1, self.w_broadcast)
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer("coord_map_const", coord_map)

    def parse_w_h(self):
        # This could be a string with expression (e.g. "32+8")
        if not isinstance(self.w_broadcast, int):
            self.w_broadcast = eval(self.w_broadcast)  # type: ignore
            assert isinstance(self.w_broadcast, int)
        if not isinstance(self.h_broadcast, int):
            self.h_broadcast = eval(self.h_broadcast)  # type: ignore
            assert isinstance(self.h_broadcast, int)

    def forward(self, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        z_tiled = (
            z.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, z.shape[1], self.h_broadcast, self.w_broadcast)
        )
        coord_map = self.coord_map_const.expand(
            batch_size, 2, self.h_broadcast, self.w_broadcast
        )
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result

MANDATORY_FIELDS = [
    "loss",  # training loss
    "mask",  # masks for all slots (incl. background if any)
    "slot",  # raw slot reconstructions for all slots (incl. background if any)
    "representation",  # slot representations (only foreground, if applicable)
]

@dataclass(eq=False, repr=False)
class BaseModel(nn.Module):
    name: str
    width: int
    height: int

    # This applies only to object-centric models, but must always be defined.
    num_slots: int

    def __post_init__(self):
        # Run the nn.Module initialization logic before we do anything else. Models
        # should call this post-init at the beginning of their post-init.
        super().__init__()

    @property
    def num_representation_slots(self) -> int:
        """Number of slots used for representation.

        By default, it is equal to the number of slots, but when possible we can
        consider only foreground slots (e.g. in SPACE).
        """
        return self.num_slots

    # @property
    # @abstractmethod
    # def slot_size(self) -> int:
    #     """Representation size per slot.

    #     This does not apply to models that are not object-centric, but they should still
    #     define it in the most sensible possible way.
    #     """
    #     ...


class Monet(BaseModel):
    def __init__(
        self,
        latent_size: int,
        num_blocks_unet: int,
        beta_kl: float,
        gamma: float,
        encoder_params: DictConfig,
        decoder_params: DictConfig,
        input_channels: int = 3,
        bg_sigma: float = 0.09,
        fg_sigma: float = 0.11,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ):
        # 필드 초기화
        self.latent_size = latent_size
        self.num_blocks_unet = num_blocks_unet
        self.beta_kl = beta_kl
        self.gamma = gamma
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.input_channels = input_channels
        self.bg_sigma = bg_sigma
        self.fg_sigma = fg_sigma
        self.prior_mean = prior_mean
        self.prior_std = prior_std

        # BaseModel 초기화
        super().__init__()

        # 수동으로 post init 작업 수행
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()
        self.attention = AttentionNet(self.input_channels, self.num_blocks_unet)
        self.encoder_params.update(width=self.width, height=self.height)
        self.encoder = EncoderNet(**self.encoder_params)
        self.decoder = BroadcastDecoderNet(**self.decoder_params)
        self.prior_dist = dists.Normal(self.prior_mean, self.prior_std)

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def forward(self, x: Tensor) -> dict:
        log_masks = self._attention_process(x)
        masks = log_masks.exp()
        zs, kl_zs, slot_means = self._encode(x, log_masks)
        slots, masks_pred = self._decode(zs)
        neg_log_pxs = self._compute_likelihood(x, slots, log_masks)
        mask_kl = self._compute_mask_kl(masks, masks_pred)
        loss = neg_log_pxs + self.beta_kl * kl_zs + self.gamma * mask_kl

        return {
            "loss": loss,
            "mask": masks.unsqueeze(2),
            "slot": slots,
            "representation": slot_means,
            "z": zs,
            "neg_log_p_x": neg_log_pxs,
            "kl_mask": mask_kl,
            "kl_latent": kl_zs,
            "mask_pred": masks_pred,
        }

    def _attention_process(self, x: Tensor) -> Tensor:
        scope_shape = list(x.shape)
        scope_shape[1] = 1
        log_scope = torch.zeros(scope_shape, device=x.device)
        log_masks = []
        for i in range(self.num_slots - 1):
            # Mask and scope: (B, 1, H, W)
            log_mask, log_scope = self.attention(x, log_scope)
            log_masks.append(log_mask)
        log_masks.append(log_scope)
        log_masks = torch.cat(log_masks, dim=1)  # (B, num slots, H, W)
        return log_masks

    def _compute_likelihood(
        self, x: Tensor, slots: Tensor, log_masks: Tensor
    ) -> Tensor:
        sigma = torch.full(
            [
                self.num_slots,
            ],
            self.fg_sigma,
            device=x.device,
        )
        sigma[0] = self.bg_sigma
        sigma = sigma.reshape([1, self.num_slots, 1, 1, 1])
        dist = dists.Normal(slots, sigma)
        log_pxs_masked = log_masks.unsqueeze(2) + dist.log_prob(x.unsqueeze(1))

        # Global negative log likelihood p(x|z): scalar.
        neg_log_pxs = -log_pxs_masked.logsumexp(dim=1).mean(dim=0).sum()

        return neg_log_pxs

    def _compute_mask_kl(self, masks: Tensor, masks_pred: Tensor) -> Tensor:
        bs = len(masks)
        d_masks = self._make_mask_distribution(masks)
        d_masks_pred = self._make_mask_distribution(masks_pred)
        kl_masks = dists.kl_divergence(d_masks, d_masks_pred)
        kl_masks = kl_masks.sum() / bs
        return kl_masks

    @staticmethod
    def _make_mask_distribution(masks: Tensor) -> dists.Distribution:
        flat_masks = masks.permute(0, 2, 3, 1).flatten(0, 2)  # (B*H*W, n_slots)
        flat_masks = flat_masks.clamp_min(1e-5)
        d_masks = dists.Categorical(probs=flat_masks)
        return d_masks

    def _encode(self, x: Tensor, log_masks: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # x: (B, 3, H, W) -> (B, num slots, 3, H, W)
        # log_masks: (B, num slots, H, W) -> (B, num slots, 1, H, W)
        x = x.unsqueeze(1).repeat(1, self.num_slots, 1, 1, 1)
        log_masks = log_masks.unsqueeze(2)

        # Encoder input: (B * num slots, RGB+mask, H, W).
        encoder_input = torch.cat([x, log_masks], 2).flatten(0, 1)

        # Encode and reshape parameters to (B, num slots, latent dim).
        mean, log_sigma = self.encoder(encoder_input).chunk(2, dim=1)
        sigma = log_sigma.exp()
        mean = mean.unflatten(0, [x.shape[0], self.num_slots])  # (B, num_slots, D)
        sigma = sigma.unflatten(0, [x.shape[0], self.num_slots])  # (B, num_slots, D)

        # Return mean, sample, and KL.
        latent_normal = dists.Normal(mean, sigma)
        kl_z = dists.kl_divergence(latent_normal, self.prior_dist)
        kl_z = kl_z.sum(dim=[1, 2]).mean(0)  # sum over latent dimensions and slots
        z = latent_normal.rsample()  # (B, num_slots, D)
        return z, kl_z, mean

    def _decode(self, zs: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = zs.shape[0]
        zs = zs.flatten(0, 1)  # (B * num slots, D)
        decoder_output = self.decoder(zs)

        # (B * num slots, 3, H, W)
        slots_recon = decoder_output[:, :3].sigmoid()
        # (B, num slots, 3, H, W)
        slots_recon = slots_recon.unflatten(0, [batch_size, self.num_slots])

        # (B * num slots, 1, H, W)
        mask_pred = decoder_output[:, 3:]
        # (B, num slots, H, W)
        mask_pred = mask_pred.unflatten(0, [batch_size, self.num_slots]).squeeze(2)
        mask_pred = mask_pred.softmax(dim=1)

        return slots_recon, mask_pred

class AttentionNet(nn.Module):
    def __init__(self, input_channels: int, num_blocks: int):
        super().__init__()
        self.unet = UNet(input_channels, num_blocks)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor, log_scope: Tensor) -> Tuple[Tensor, Tensor]:
        inp = torch.cat((x, log_scope), 1)
        logits = self.unet(inp)
        log_alpha = self.log_softmax(logits)
        log_mask = log_scope + log_alpha[:, 0:1]
        log_scope = log_scope + log_alpha[:, 1:2]
        return log_mask, log_scope
