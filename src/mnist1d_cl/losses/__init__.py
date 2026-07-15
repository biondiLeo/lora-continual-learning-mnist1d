"""Loss functions: knowledge distillation (KL teacher||student)."""

from .kd import kd_loss

__all__ = ["kd_loss"]
