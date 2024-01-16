from typing import List, Optional

from parameters.xor.xor_parameters import XorParams


class XorFParams(XorParams):
    start_dimensions: List[int] = [6, 6, 5, 3, 2]
    num_layers: int = 4
    task_version_name: Optional[str] = 'xor_f'
