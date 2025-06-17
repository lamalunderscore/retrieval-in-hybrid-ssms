"""Utility functions for dejavu implementation."""

from typing import Any, Literal

import torch


class AttentionRecorder(torch.nn.Module):
    """A class that implements the recording of Tensors, either in a Module, or a Model.

    Attributes:
        record_mode (Literal["first", "last", "all"]): Which record mode is used.
        data (List[torch.Tensor | Any] | torch.Tensor | None): The recorded data.

    """

    def __init__(self, name: str, record_mode: Literal["first", "last", "all"] = "first"):
        super().__init__()
        self.name = name
        self.record_mode = record_mode

        self._clear()

        # Map enum to forward methods
        self._forward_methods = {
            "all": self._forward_all,
            "last": self._forward_last,
            "first": self._forward_first,
        }

    def _clear(self):
        # Initialize data based on mode
        self.data: list[torch.Tensor | Any] | torch.Tensor | None = None  # "first" and "last"
        if self.record_mode == "all":
            self.data = []

    def get(self) -> list[torch.Tensor | None] | torch.Tensor | None:
        """Get and clear recorder data.

        Raises:
            TypeError: If self.data is of an unknown type.

        Returns:
            list[torch.Tensor | None] | torch.Tensor | None: A copy of the stored data.

        """
        if isinstance(self.data, list):
            return_data = [tensor.clone() for tensor in self.data]
        elif isinstance(self.data, torch.Tensor):
            return_data = self.data.clone()
        elif self.data is None:
            print(f"Warning, recorder data of layer {self.name} is empty, returning None.")
            return None
        else:
            raise TypeError(
                f"Expected self.data to be of type list, torch.Tensor or None. Got {type(self.data)} instead."
            )

        # ensure tensors
        if isinstance(return_data, list):
            for tensor in return_data:
                assert isinstance(tensor, torch.Tensor), "Tensor not of type Tensor."
        else:
            assert isinstance(return_data, torch.Tensor), "Tensor not of type Tensor."

        return return_data

    def forward(self, x):
        """Dispatch to appropriate forward method based on record mode."""
        self._forward_methods[self.record_mode](x)

    def _forward_all(self, x):
        """Record all inputs in a list."""
        assert isinstance(self.data, list) and isinstance(x, torch.Tensor), (
            f"Need self.data to be `list` and x to be `torch.Tensor`, but got {type(self.data)} and {type(x)}."
        )
        print("DEBUG: in recorder._forward_all")
        self.data.append(x.detach().cpu())

    def _forward_first(self, x):
        """Record only the first input."""
        assert isinstance(x, torch.Tensor), f"Need x to be `torch.Tensor`, but got {type(x)}."
        print("DEBUG: in recorder._forward_first")
        if self.data is None:
            print(f"DEBUG: {self.name}.data is None, so save tensor")
            self.data = x.detach().cpu()
        else:
            print(f"DEBUG: {self.name}.data is not None, so skip.")

    def _forward_last(self, x):
        """Record only the last input."""
        print("DEBUG: in recorder._forward_last")
        assert isinstance(x, torch.Tensor), f"Need x to be `torch.Tensor`, but got {type(x)}."
        self.data = x.detach().cpu()


__all__ = ("AttentionRecorder",)
