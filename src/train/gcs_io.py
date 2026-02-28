from google.cloud import storage


def upload_to_gcs(
    local_file_path: str,
    bucket_name: str,
    destination: str,
):
    client = storage.Client()

    bucket = client.bucket(bucket_name)

    blob = bucket.blob(destination)

    blob.upload_from_filename(local_file_path)

    print(f"File {local_file_path} uploaded to gs://{bucket_name}/{destination}")