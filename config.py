#YouTube Faces (YTF), 1: FaceScrub (FS), 2: VGGFace2 (VGGF), 3: CASIA-WebFace (CW)

cifar = dict(
    NB_CLS=10,
    input_size=224)

cifar32 = dict(
    Gallery_img_dir='./data/cifar',
    Gallery_txt_dir='./data/cifar_Train.txt',
    Query_img_dir='./data/cifar',
    Query_txt_dir='./data/cifar_Query.txt',
    NB_CLS=10,
    input_size=32)

ImageNet32 = dict(
    Train_img_dir='./data/ImageNet_32',
    Train_txt_dir='./data/ImageNet32.txt',
    NB_CLS=1000,
    input_size=32)
