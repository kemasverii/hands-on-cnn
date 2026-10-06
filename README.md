# CNN From Scratch - Praktikum

Setiap file dapat dijalankan secara mandiri.

## CIFAR-10
File:
- 01_lenet_cifar10.py
- 03_vgg_cifar10.py
- 05_resnet_cifar10.py
- 07_mobilenet_cifar10.py
- 09_convnext_cifar10.py

Dataset akan di-download otomatis.
Default:
- train: 5000 sampel
- validation: 1000 sampel
- epoch: 5
- image size: 64x64

Contoh:
python 05_resnet_cifar10.py

## ImageNet subset
File:
- 02_alexnet_imagenet.py
- 04_googlenet_imagenet.py
- 06_densenet_imagenet.py
- 08_efficientnet_imagenet.py

Gunakan struktur:
imagenet/
  train/
    class_1/
    class_2/
  val/
    class_1/
    class_2/

Tidak harus memakai seluruh ImageNet-1K. Subset beberapa kelas dapat digunakan untuk praktikum.
Default:
- train: maksimum 4000 sampel
- validation: maksimum 800 sampel
- epoch: 5
- image size: 128x128

Contoh:
python 08_efficientnet_imagenet.py
