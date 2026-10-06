# CNN From Scratch - Praktikum

Setiap file dapat dijalankan secara mandiri.

## CIFAR-10
File:
- 01_lenet_cifar10.ipynb
- 03_vgg_cifar10.ipynb
- 05_resnet_cifar10i.ipynb
- 07_mobilenet_cifar10.ipynb
- 09_convnext_cifar10.ipynb

Dataset akan di-download otomatis.
Default:
- train: 5000 sampel
- validation: 1000 sampel
- epoch: 5
- image size: 64x64


## ImageNet subset
File:
- 02_alexnet_imagenet.ipynb
- 04_googlenet_imagenet.ipynb
- 06_densenet_imagenet.ipynb
- 08_efficientnet_imagenet.ipynb

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

