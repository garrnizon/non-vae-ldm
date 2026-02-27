import requests
from urllib.parse import urlencode, urlparse, parse_qs
import os
from tqdm import tqdm

def download_from_yadisk(short_url: str, target_dir: str = '.', filename: str = None):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

    final_url = base_url + urlencode(dict(public_key=short_url))
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Extract filename from the download URL if not provided
    if filename is None:
        qs = parse_qs(urlparse(download_url).query)
        filename = qs.get('filename', [os.path.basename(urlparse(short_url).path)])[0]

    print('Starting downloading file:', filename)

    # Stream the response to track progress
    download_response = requests.get(download_url, stream=True)
    total_size = int(download_response.headers.get('content-length', 0))

    target_file = os.path.join(target_dir, filename)
    with open(target_file, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in download_response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print('Finished downloading file:', filename)
