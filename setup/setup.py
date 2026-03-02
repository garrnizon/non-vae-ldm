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
    
    imagenet_dir = f'{data_dir}/imagenet'
    os.makedirs(imagenet_dir)

    for class_ in config['imagenet_classes']:
        os.system(f'wget https://image-net.org/data/winter21_whole/{class_}.tar')
    
    for class_ in config['imagenet_classes']:
        class_dir = f'{imagenet_dir}/{class_}'
        os.makedirs(class_dir)
        os.system(f'tar -xf {class_}.tar -C {class_dir}')
        os.remove(f'{class_}.tar')
    
    unzip_files_in_directory(data_dir)
