from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pandas as pd

# Grab images metadata if it is available.


def get_image_metadata(image_path):

    image = Image.open(image_path)

    metadata = {
        "filename": image.filename,
        "camera_make": None,
        "camera_model": None,
        "date_time": None,
        "latitude": None,
        "longitude": None
    }

    exif = image.getexif()

    if not exif:
        return metadata

    # -------------------------
    # Basic EXIF information
    # -------------------------

    for tag_id, value in exif.items():

        tag_name = TAGS.get(
            tag_id,
            tag_id
        )

        if tag_name == "Make":
            metadata["camera_make"] = value

        elif tag_name == "Model":
            metadata["camera_model"] = value

        elif tag_name == "DateTime":
            metadata["date_time"] = value

    # -------------------------
    # GPS information
    # -------------------------

    try:
        gps_info = exif.get_ifd(0x8825)
    except Exception:
        gps_info = None

    if not gps_info:
        return metadata

    gps_data = {}

    for key, value in gps_info.items():

        tag_name = GPSTAGS.get(
            key,
            key
        )

        gps_data[tag_name] = value

    if (
        "GPSLatitude" not in gps_data
        or "GPSLongitude" not in gps_data
    ):
        return metadata

    latitude_ref = gps_data.get(
        "GPSLatitudeRef"
    )

    longitude_ref = gps_data.get(
        "GPSLongitudeRef"
    )

    if not latitude_ref or not longitude_ref:
        return metadata

    metadata["latitude"] = convert_to_decimal(
        gps_data["GPSLatitude"],
        latitude_ref
    )

    metadata["longitude"] = convert_to_decimal(
        gps_data["GPSLongitude"],
        longitude_ref
    )

    return metadata

# Convert to GPS coordinates

def convert_to_decimal(coordinates, reference):

    degrees = float(coordinates[0])
    minutes = float(coordinates[1])
    seconds = float(coordinates[2])

    decimal = (
        degrees
        + minutes / 60
        + seconds / 3600
    )

    if reference in ["S", "W"]:
        decimal *= -1

    return decimal


# -------------------------
# Test using test dataset
# -------------------------

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    all_data = pd.read_csv(
        "train_ids_labels.csv"
    )

    # Recreate the same 80/10/10 split
    train_data, temp_data = train_test_split(
        all_data,
        test_size=0.2,
        stratify=all_data["Label"],
        random_state=42
    )

    validation_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        stratify=temp_data["Label"],
        random_state=42
    )

    # Select one random image from the test set
    random_row = test_data.sample(
        n=1
    ).iloc[0]

    image_id = random_row["Image_ID"]
    label = random_row["Label"]

    image_path = (
        "all_data/"
        + image_id
        + ".JPG"
    )

    metadata = get_image_metadata(
        image_path
    )

    print("--------------------------------")
    print("TEST DATA IMAGE")
    print("--------------------------------")

    print(
        "Image ID:",
        image_id
    )

    print(
        "Actual label:",
        label
    )

    print(
        "Filename:",
        metadata["filename"]
    )

    print(
        "Camera:",
        metadata["camera_make"],
        metadata["camera_model"]
    )

    print(
        "Date taken:",
        metadata["date_time"]
    )

    print(
        "Latitude:",
        metadata["latitude"]
    )

    print(
        "Longitude:",
        metadata["longitude"]
    )
