from typing import List, Optional

from parameters.retina.retina_params import RetinaParams


class RetinaCParams(RetinaParams):
    start_dimensions: List[int] = [6, 5, 5, 2]
    num_layers: int = 3
    task_version_name: Optional[str] = 'retina_c'

