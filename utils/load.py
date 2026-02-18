from yfile import download_from_yadisk

download_from_yadisk(
    'https://disk.yandex.ru/d/Hbcp2fYTWLpdpg',
    'svg_autoencoder_dinov3s16p_vit-s_epoch40.ckpt',
    './data/'
)

download_from_yadisk(
    'https://disk.yandex.ru/d/4o0wqiQaJ45Jyw',
    'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
    './data/'
)