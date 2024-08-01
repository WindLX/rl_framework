from typing import Optional, Union

from pydantic import BaseModel, model_validator

from ...utils.error import SharedACError


class ACConfig(BaseModel):
    """AC Configurations"""

    # is actor and critic share the same model
    is_model_shared: bool
    # save_path
    save_path: Optional[str] = None
    # clip grad norm
    clip_grad_norm: Union[Optional[float], dict[str, Optional[float]]] = None

    @model_validator(mode="after")
    def validate_clip_grad_norm(self):
        if self.is_model_shared:
            if isinstance(self.clip_grad_norm, dict):
                raise SharedACError(
                    "clip_grad_norm should be a float when is_model_shared is True"
                )
            return self
        else:
            if not isinstance(self.clip_grad_norm, dict) and (
                "actor" not in self.clip_grad_norm.keys()
                or "critic" not in self.clip_grad_norm.keys()
            ):
                raise SharedACError(
                    "clip_grad_norm should be a dict when is_model_shared is False"
                )
            return self


class ACEvalConfig(BaseModel):
    """Evaluation Configurations"""

    is_model_shared: bool = True
    # save path
    save_path: Optional[str] = None
    # is_render
    is_render: bool = True
