import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import train_utils.distributed_utils as utils

from src import UNet
from src.ddu_net import *
from src.LMFR_net import *
from src.B import *
from src.B_ICB import *
from src.B_ICB_LMAFF import *

from my_dataset_drive import *

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 2  # exclude background
    weights_path = "save_weights/{weightfile_name}.pth"
    test_img_path = "./DRIVE"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(test_img_path), f"weights {test_img_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UNet(in_channels=3, num_classes=classes, base_c=64)
    # model = DDU_Net()
    # model = LMFR_Net()
    # model = LMFR_Net_B()
    # model = LMFR_Net_B_ICB()
    model = LMFR_Net_B_ICB_LMAFF()

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # from pil image to tensor and normalize
    data_transform_1 = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    data_transform_2 = transforms.Compose([transforms.ToTensor()])

    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    mean_acc = 0
    mean_sp = 0
    mean_auc = 0

    for i in range(5):
        origin_img, label, roi, name = Test_data(test_img_path, i)
        print(name)
        # origin_img, label, roi = Test_data_Chase(test_img_path, i)
        img = data_transform_1(origin_img)
        img = torch.unsqueeze(img, dim=0)
        label = data_transform_2(label).to(device)

        roi = np.array(roi)

        model.eval()  
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            confmat = utils.ConfusionMatrix(classes)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            confmat.update(label.flatten(), output['out'].argmax(1).flatten())
            test_info = str(confmat)
            print(test_info)

            acc = test_info[16:21]
            f1 = test_info[-5:]
            recall = test_info[-15:-10]
            precision = test_info[-29:-24]
            sp = test_info[-47:-42]
            auc = test_info[-65:-60]
            mean_acc += float(acc)
            mean_precision += float(precision)
            mean_f1 += float(f1)
            mean_recall += float(recall)
            mean_sp += float(sp)
            mean_auc += float(auc)

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction[prediction == 1] = 255
            prediction[roi == 0] = 0
            mask = Image.fromarray(prediction)
            # mask.show()

    mean_acc = mean_acc / 5
    mean_f1 = mean_f1 / 5
    mean_recall = mean_recall / 5
    mean_precision = mean_precision / 5
    mean_sp = mean_sp / 5
    mean_auc = mean_auc / 5

    print('mean_acc:', mean_acc)
    print('mean_f1:', mean_f1)
    print('mean_recall:', mean_recall)
    print('mean_precision:', mean_precision)
    print('mean_sp:', mean_sp)
    print('mean_auc:', mean_auc)


if __name__ == '__main__':
    main()
