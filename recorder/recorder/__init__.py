"""Import structure for recorder package."""

from . import attention


AttentionRecorder = attention.AttentionRecorder


__all__ = ("AttentionRecorder",)

# Prevents from accessing anything except the exported symbols
try:
    del torch, attention  # type: ignore
except NameError:
    pass
