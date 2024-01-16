from typing import List, Optional

from parameters.xor.xor_parameters import XorParams


class XorDParams(XorParams):
    start_dimensions: List[int] = [6, 8, 2]
    num_layers: int = 2
    task_version_name: Optional[str] = 'xor_d'
