import yaml
import os

from unzip import unzip_files_in_directory
from yfile import download_from_yadisk

if __name__ == '__main__':
    with open('setup/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data_dir']
    os.makedirs(data_dir)
    for file_url in config['yadisk']:
        download_from_yadisk(file_url, data_dir)

    unzip_files_in_directory(data_dir)
