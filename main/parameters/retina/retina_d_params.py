from typing import List, Optional

from parameters.retina.retina_params import RetinaParams


class RetinaDParams(RetinaParams):
    start_dimensions: List[int] = [6, 4, 4, 3, 3, 2]
    num_layers: int = 5
    task_version_name: Optional[str] = 'retina_d'
    num_epochs: int = 1700
