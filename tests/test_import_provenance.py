from pathlib import Path

import project
import scripts


def test_repository_packages_come_from_this_checkout() -> None:
    repository = Path(__file__).resolve().parents[1]

    for package in (project, scripts):
        imported = Path(package.__file__).resolve()
        assert imported.is_relative_to(repository), (
            f"pytest imported {package.__name__} from another checkout: {imported}"
        )
