"""Identical benchmark observation on both main and candidate adapters."""

from __future__ import annotations

import time
from typing import Any

from sie_server.adapters.cross_encoder import CrossEncoderAdapter


class ObservedCrossEncoder(CrossEncoderAdapter):
    def load(self, device: str) -> None:
        self.batch_observations = []
        self.selection_observations = []
        self.observed_selection = None
        self.observation_phase = "load"
        self.capture_logits = True
        self.active_forward = False
        self.mutation_overlap = False
        self.forward_observations = []
        self.fetch_observations = []
        super().load(device)
        assert self._model is not None
        self._observer_handle = self._model.model.register_forward_hook(self._observe_forward, with_kwargs=True)

    def _observe_forward(self, model: Any, _args: Any, kwargs: Any, result: Any) -> None:
        enabled = []
        if self.capture_logits:
            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and not getattr(module, "disable_adapters", False):
                    enabled.append({"module": name, "adapters": list(module.active_adapters)})
        inputs = kwargs.get("input_ids")
        row = {
            "selection": self.observed_selection,
            "enabled_adapters": enabled,
            "input_shape": list(inputs.shape) if inputs is not None else None,
        }
        if self.capture_logits:
            row["logits"] = result.logits.detach().float().cpu().tolist()
        self.forward_observations.append(row)

    def set_active_lora(self, lora_name: str | None) -> None:
        if self.active_forward:
            self.mutation_overlap = True
        super().set_active_lora(lora_name)
        self.observed_selection = lora_name
        self.selection_observations.append({"time_ns": time.time_ns(), "selection": lora_name})

    def score_pairs(self, queries: list, docs: list, **kwargs: Any) -> Any:
        started = time.time_ns()
        first_forward = len(self.forward_observations)
        self.active_forward = True
        try:
            result = super().score_pairs(queries, docs, **kwargs)
            self.batch_observations.append(
                {
                    "start_ns": started,
                    "end_ns": time.time_ns(),
                    "selection": self.observed_selection,
                    "phase": self.observation_phase,
                    "queries": [item.text for item in queries],
                    "documents": [item.text for item in docs],
                    "item_ids": [item.id for item in docs],
                    "scores": result.scores.tolist(),
                    "input_token_counts": result.input_token_counts,
                    "forwards": self.forward_observations[first_forward:],
                }
            )
            return result
        finally:
            self.active_forward = False

    def load_lora(self, *args: Any, **kwargs: Any) -> int:
        if self.active_forward:
            self.mutation_overlap = True
        return super().load_lora(*args, **kwargs)

    def fetch_lora(self, *args: Any, **kwargs: Any) -> str:
        self.fetch_observations.append(
            {"time_ns": time.time_ns(), "name": args[0] if args else kwargs.get("lora_name")}
        )
        return super().fetch_lora(*args, **kwargs)

    def unload_lora(self, *args: Any, **kwargs: Any) -> None:
        if self.active_forward:
            self.mutation_overlap = True
        super().unload_lora(*args, **kwargs)

    def unload(self) -> None:
        handle = getattr(self, "_observer_handle", None)
        if handle is not None:
            handle.remove()
            self._observer_handle = None
        super().unload()
