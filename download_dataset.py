"""
Download NSL-KDD Dataset
Automatically downloads training and test datasets
"""

import os
import urllib.request
import sys

DATASET_DIR = "ml/dataset"
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"

FILES = {
    "KDDTrain+.txt": BASE_URL + "KDDTrain+.txt",
    "KDDTest+.txt": BASE_URL + "KDDTest+.txt",
}


def download_file(url, filepath):
    """Download file with progress"""
    print(f"Downloading {os.path.basename(filepath)}...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\r  Progress: {percent:.1f}% [{downloaded}/{total_size} bytes]")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filepath, reporthook=progress)
        print()  # New line after progress
        print(f"  ✓ Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("  NSL-KDD Dataset Downloader")
    print("=" * 60)

    # Create dataset directory
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"\n✓ Created directory: {DATASET_DIR}")

    # Download files
    print(f"\nDownloading to: {os.path.abspath(DATASET_DIR)}/")
    print()

    success_count = 0
    for filename, url in FILES.items():
        filepath = os.path.join(DATASET_DIR, filename)

        # Check if already exists
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✓ {filename} already exists ({file_size} bytes)")
            print(f"  Skipping download. Delete file to re-download.")
            success_count += 1
            continue

        # Download
        if download_file(url, filepath):
            success_count += 1

        print()

    # Summary
    print("=" * 60)
    if success_count == len(FILES):
        print("✅ All files downloaded successfully!")
        print("\nNext steps:")
        print("  1. Run training: python ml/train.py")
        print("  2. Start backend: python backend/main.py")
    else:
        print(f"⚠ Downloaded {success_count}/{len(FILES)} files")
        print("\nYou can manually download from:")
        print("  https://www.unb.ca/cic/datasets/nsl.html")

    print("=" * 60)


if __name__ == "__main__":
    main()
