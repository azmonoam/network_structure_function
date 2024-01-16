from typing import List, Optional

from parameters.retina.retina_params import RetinaParams


class RetinaFParams(RetinaParams):
    start_dimensions: List[int] = [6, 4, 3, 2]
    num_layers: int = 3
    task_version_name: Optional[str] = 'retina_f'
