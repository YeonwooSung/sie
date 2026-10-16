"""Benchmark-owned observation hook; inference uses the production adapter."""

from pathlib import Path

from sie_server.adapters.sglang.generation import SGLangGenerationAdapter


class ObservedGenerationAdapter(SGLangGenerationAdapter):
    def _compat_pythonpath_entries(self) -> tuple[str, ...]:
        source = Path(__file__).resolve()
        return (str(source.parent / "_hooks"), str(source.parents[3]), *super()._compat_pythonpath_entries())
