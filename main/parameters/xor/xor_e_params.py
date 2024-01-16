from typing import List, Optional

from parameters.xor.xor_parameters import XorParams


class XorEParams(XorParams):
    start_dimensions: List[int] = [6, 5, 4, 3, 3, 2, 2]
    num_layers: int = 6
    task_version_name: Optional[str] = 'xor_e'
