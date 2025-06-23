import json
from typing import Dict, Optional

from src.apriltags.utils.apriltag import Apriltag


def load_fmap_file(fmap_file_path: str) -> Dict[int, Apriltag]:
    """Load and parse an fmap apriltag map file into Apriltag objects keyed by id.

    Args:
        fmap_file_path (str): Path to the fmap file.

    Returns:
        Dict[int, Apriltag]: Dictionary of Apriltag objects keyed by their id.
    """
    with open(fmap_file_path, "r", encoding="utf-8") as fmap_file:
        fmap_content = fmap_file.read()
    fmap_data = json.loads(fmap_content)
    apriltag_dict: Dict[int, Apriltag] = {}
    for fiducial in fmap_data.get("fiducials", []):
        apriltag = Apriltag(
            tag_id=fiducial["id"],
            family=fiducial["family"],
            size=fiducial["size"],
            transform=fiducial["transform"],
            unique=fiducial["unique"],
            field_length=fmap_data["fieldlength"],
            field_width=fmap_data["fieldwidth"],
        )
        apriltag_dict[fiducial["id"]] = apriltag
    return apriltag_dict


def get_apriltag_data(fmap_file_path: str, tag_id: int) -> Optional[Apriltag]:
    """Get apriltag data for a specific tag ID from an fmap file.

    Args:
        fmap_file_path (str): Path to the fmap file.
        tag_id (int): The ID of the apriltag to retrieve.

    Returns:
        Optional[Apriltag]: The Apriltag object if found, None otherwise.
    """
    apriltag_dict = load_fmap_file(fmap_file_path)
    return apriltag_dict.get(tag_id)
