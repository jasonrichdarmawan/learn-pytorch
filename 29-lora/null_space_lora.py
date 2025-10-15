from peft import PeftModel, PeftConfig, LoraConfig
from peft.tuners.lora import LoraModel, LoraLayer, Linear, Embedding, Conv2d
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import PeftType
from peft.utils.other import transpose
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from typing import Union
import warnings
from itertools import chain
import re


class NullSpaceLinear(Linear):

    def __init__(
        self,
        base_layer: nn.Module,
        P: torch.Tensor,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace store weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        self.P = P
        super().__init__(
            base_layer=base_layer,
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            is_target_conv_1d_layer=is_target_conv_1d_layer,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self._unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # P is a projection matrix, so P.T == P
                    # U \Lambda U^T = SVD(K_0 K_0^T)
                    # P = UU^T
                    result = result + (lora_B(lora_A(dropout(x) @ self.P))) * scaling
                else:
                    raise NotImplementedError("DoRa is not implemented yet.")

            result = result.to(torch_result_dtype)

        return result

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        output_tensor = (
            transpose(weight_B @ weight_A @ self.P, fan_in_fan_out=self.fan_in_fan_out)
            * self.scaling[adapter]
        )

        return output_tensor


class NullSpaceLoraModel(LoraModel):
    def __init__(
        self,
        model: PreTrainedModel,
        config: PeftConfig,
        adapter_name: str,
        P_map: dict[str, torch.Tensor],
    ):
        self.P_map = P_map
        super().__init__(model=model, config=config, adapter_name=adapter_name)

    def _create_and_replace(
        self,
        lora_config: LoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(
            chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys())
        )
        target_name_key = next(
            filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys),
            current_key,
        )
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "P": self.P_map[current_key],
        }

        # quant_method is not implemented yet

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_module = self._create_new_module(
                lora_config, adapter_name, target, **kwargs
            )
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        lora_config: LoraConfig, adapter_name: str, target: nn.Module, **kwargs
    ):
        dispatchers = []

        # dispatch_bnb_8bit, dispatch_bnb_4bit
        # dispatch_aqlm, dispatch_awq, dispatch_gptq,
        # dispatch_megatron not implemented yet

        dispatchers.extend([dispatcher_default])

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target, adapter_name, lora_config=lora_config, **kwargs
            )
            if new_module is not None:
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module


def dispatcher_default(
    target: nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(
            base_layer=target, adapter_name=adapter_name, **embedding_kwargs
        )
    elif isinstance(target_base_layer, nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(base_layer=target, adapter_name=adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        P = kwargs.pop("P")
        new_module = NullSpaceLinear(
            base_layer=target, P=P, adapter_name=adapter_name, **kwargs
        )
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        P = kwargs.pop("P")
        new_module = NullSpaceLinear(
            base_layer=target,
            P=P,
            adapter_name=adapter_name,
            is_target_conv_1d_layer=True,
            **kwargs,
        )

    return new_module


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: NullSpaceLoraModel,
}


class NullSpacePeftModel(PeftModel):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        P_map: dict[str, torch.Tensor],
        adapter_name: str = "default",
    ) -> None:
        nn.Module.__init__(self)  # Only initialize the nn.Module part
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config)
        else:
            self._peft_config = None
            cls = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
            kwargs = {}
            if peft_config.peft_type == PeftType.LORA:
                kwargs["P_map"] = P_map
            self.base_model = cls(
                model, {adapter_name: peft_config}, adapter_name, **kwargs
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(
            self.base_model.config, "pretraining_tp"
        ):
            self.base_model.config.pretraining_tp = 1
