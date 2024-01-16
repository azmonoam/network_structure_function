from typing import Optional, List, Callable

from parameters.retina.retina_params import RetinaParams


class RetinaByDim(RetinaParams):
    def __init__(
            self,
            num_layers: int,
            start_dimensions: List[int],
            num_epochs: int = 10000,
            by_epochs: bool = True,
            rule: Optional[Callable] = None,
            task_base_folder_name: Optional[str] = None,
    ):
        self.num_layers: int = num_layers
        self.task_version_name: Optional[str] = self._get_task_version_name(
            num_layers=num_layers,
            start_dimensions=start_dimensions,
        )
        self.task_base_folder_name = task_base_folder_name
        self.num_epochs = num_epochs
        self.by_epochs = by_epochs
        super().__init__(
            start_dimensions=start_dimensions,
            rule=rule,
        )

    @property
    def output_folder(cls) -> str:
        if cls.by_epochs:
            return f"{cls.teach_arch_base_path}/{cls.num_epochs}_ep/teach_archs_results"
        return f"{cls.teach_arch_base_path}/teach_archs_results"

    def _get_num_epochs(self) -> int:
        return self.num_epochs

    @staticmethod
    def _get_task_version_name(
            num_layers: int,
            start_dimensions: List[int],
    ) -> str:
        task_version_name = f'retina_{num_layers}_layers'
        for i in range(1, num_layers):
            task_version_name += f'_{start_dimensions[i]}'
        return task_version_name
