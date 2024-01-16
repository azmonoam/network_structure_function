import random
from typing import Optional

from parameters.retina.retina_params import RetinaParams


class DynamicRetinaParams(RetinaParams):
    def __init__(
            self,
            num_layers: int,
    ):
        self.num_layers: int = num_layers
        self.task_version_name: Optional[str] = f'dynamic_retina_{num_layers}_layers'
        self.num_epochs = 10000
        super().__init__()

    def _get_start_dimensions(cls):
        start_dimensions = [cls.input_dim]
        for layer in range(cls.num_layers - 1):
            start_dimensions.append(
                random.randint(min(cls.input_dim, cls.output_dim), max(cls.input_dim, cls.output_dim))
            )
        start_dimensions.append(cls.output_dim)
        return start_dimensions

    def _get_num_epochs(self) -> int:
        return self.num_epochs
