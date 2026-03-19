import boto3
from botocore.exceptions import ClientError

# Ask someone for the below
ACCESS_KEY = ""
SECRET_KEY = ""

REGION     = "atl1"
ENDPOINT   = f"https://{REGION}.digitaloceanspaces.com"
BUCKET     = "peersai"
S3_KEY = "gamma_complete_dataset.jsonl.gz"

s3 = boto3.client(
    "s3",
    region_name=REGION,
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

# Where to save locally
DOWNLOAD_PATH = r"./gamma_complete_dataset_downloaded.jsonl.gz"

try:
    print(f"Downloading {S3_KEY} → {DOWNLOAD_PATH}...")

    s3.download_file(
        Bucket=BUCKET,
        Key=S3_KEY,
        Filename=DOWNLOAD_PATH
    )

    print("✅ Download complete!")

except ClientError as e:
    print("❌ Download failed:")
    print(e.response["Error"]["Code"], e.response["Error"]["Message"])

