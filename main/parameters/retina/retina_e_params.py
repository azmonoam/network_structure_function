from typing import List, Optional

from parameters.retina.retina_params import RetinaParams


class RetinaEParams(RetinaParams):
    start_dimensions: List[int] = [6, 2]
    num_layers: int = 1
    task_version_name: Optional[str] = 'retina_e'
