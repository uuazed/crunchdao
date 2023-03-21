import os
import logging

import tqdm
import requests


logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: str,
                  show_progress_bars: bool = True) -> str:
    """downloads a file and shows a progress bar. allows resuming a download"""
    file_size = 0
    req = requests.get(url, stream=True)
    req.raise_for_status()

    # Total size in bytes.
    total_size = int(req.headers.get('content-length', 0))

    if os.path.exists(dest_path):
        logger.info("target file already exists")
        file_size = os.stat(dest_path).st_size  # File size in bytes
        if file_size < total_size:
            # Download incomplete
            logger.info("resuming download")
            resume_header = {'Range': f'bytes={file_size}-'}
            req = requests.get(url, headers=resume_header, stream=True,
                               verify=False, allow_redirects=True)
        elif file_size == total_size:
            # Download complete
            logger.info("download complete")
            return dest_path
        else:
            # Error, delete file and restart download
            logger.error("deleting file and restarting")
            os.remove(dest_path)
            file_size = 0
    else:
        # File does not exist, starting download
        logger.info("starting download")

    # write dataset to file and show progress bar
    pbar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=dest_path, disable=not show_progress_bars)
    # Update progress bar to reflect how much of the file is already downloaded
    pbar.update(file_size)
    with open(dest_path, "ab") as dest_file:
        for chunk in req.iter_content(1024):
            dest_file.write(chunk)
            pbar.update(1024)
    return dest_path
