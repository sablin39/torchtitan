from .model import RAEDecoder

__all__ = ["RAEDecoder"]


def __getattr__(name: str):
    if name in {"RAEStage1Trainer", "RAEGANConfig"}:
        from .trainer import RAEGANConfig, RAEStage1Trainer

        return {"RAEStage1Trainer": RAEStage1Trainer, "RAEGANConfig": RAEGANConfig}[
            name
        ]
    if name in {"model_registry", "rae_stage1_debug", "rae_stage1_dmuon"}:
        from .config_registry import model_registry, rae_stage1_debug, rae_stage1_dmuon

        return {
            "model_registry": model_registry,
            "rae_stage1_debug": rae_stage1_debug,
            "rae_stage1_dmuon": rae_stage1_dmuon,
        }[name]
    raise AttributeError(name)
