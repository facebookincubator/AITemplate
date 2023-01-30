from typing import Any, Callable, Dict

from torch.fx.node import Target

AIT_CONVERTERS: Dict[Target, Any] = {}


def ait_converter(key: Target, enabled: bool = True) -> Callable[[Any], Any]:
    def register_converter(converter):
        AIT_CONVERTERS[key] = converter
        return converter

    def disable_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return disable_converter
