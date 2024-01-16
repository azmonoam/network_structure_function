from typing import List, Optional

from parameters.retina.retina_params import RetinaParams


class RetinaBParams(RetinaParams):
    start_dimensions: List[int] = [6, 3, 2]
    num_layers: int = 2
    task_version_name: Optional[str] = 'retina_b'
