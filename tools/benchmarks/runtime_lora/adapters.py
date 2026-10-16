"""Benchmark-owned observation hook; inference uses the production adapter."""

from pathlib import Path

from sie_server.adapters.sglang.generation import SGLangGenerationAdapter


class ObservedGenerationAdapter(SGLangGenerationAdapter):
    def _compat_pythonpath_entries(self) -> tuple[str, ...]:
        return (str(Path(__file__).parent / "_hooks"), *super()._compat_pythonpath_entries())
