from typing import List, Optional

from parameters.xor.xor_parameters import XorParams


class XorCParams(XorParams):
    start_dimensions: List[int] = [6, 6, 4, 2]
    num_layers: int = 3
    task_version_name: Optional[str] = 'xor_c'
