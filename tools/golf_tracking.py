import argparse
import os
import time
from loguru import logger
import sys
import cv2
import numpy as np
import imageio
import torch

current_script_directory = os.path.dirname(os.path.abspath(__file__))


# Add the parent directory of the current script to sys.path
parent_directory = os.path.dirname(current_script_directory)
sys.path.append(parent_directory)

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets.voc_classes import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--center", 
        default=(1130, 1020), 
        type=int,
        nargs='+',
        help="initial center point",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35, crop_region=None):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names, crop_region)
        return vis_res

def create_gif(image_list:list, gif_path:str, duration=0.3):
    """
    Create a GIF from a list of images represented by NumPy arrays.

    Parameters:
        image_list (list of ndarray): List of images (NumPy arrays).
        gif_path (str): Path to save the GIF file.
        duration (float, optional): Time duration for each frame in seconds. Default is 0.3 seconds.
    """
    
    images = [cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB) for image in image_list]
    imageio.mimsave(gif_path, images, duration=duration)

def simple_tracking(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
    detection_list = []
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        detection_list.append(result_image)
        if save_result:
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    save_tracking_file = os.path.join(save_folder, "tracking.gif")
    create_gif(detection_list, save_tracking_file)

def crop_around_center(image, center_x, center_y, crop_width, crop_height):
    """
    Crop an image around a specified center point with a given width and height.

    Parameters:
        image (ndarray): Input image.
        center_x (int): X-coordinate of the center point.
        center_y (int): Y-coordinate of the center point.
        crop_width (int): Width of the crop.
        crop_height (int): Height of the crop.

    Returns:
        ndarray: Cropped image.
        coodinates
    """
    half_width = crop_width // 2
    half_height = crop_height // 2

    # Calculate the coordinates of the top-left corner of the crop
    x1 = max(0, center_x - half_width)
    y1 = max(0, center_y - half_height)

    # Calculate the coordinates of the bottom-right corner of the crop
    x2 = min(image.shape[1], center_x + half_width)
    y2 = min(image.shape[0], center_y + half_height)

    # Perform the crop
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image, x1, y1, x2, y2

def kalman_tracking(predictor, vis_folder, path, current_time, save_result, init_center):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
    detection_list = []

    """
    Kalman Filter
    """

    Transition_Matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    Observation_Matrix = [[1, 0, 1, 0], [0, 1, 0, 1]]
    Xhat = np.zeros((4))
    P = np.zeros((4, 4))
    Xhatminus = np.zeros((4))
    Pminus = np.zeros((4, 4))
    K = np.zeros((4, 2))
    # trust the measurement result much more then estimation result
    Q = 0.1 * np.eye(4)
    R = 0.0001 * np.eye(2) 
        
    for i,image_name in enumerate(files):
        img = cv2.imread(image_name)
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = 1.0
        
        Xhatminus = np.matmul(Transition_Matrix, Xhat)
        Pminus = np.matmul(np.matmul(Transition_Matrix, P), Transition_Matrix) + Q
        if i == 0:
            center_x, center_y = init_center
        else:
            center_x, center_y = int(Xhatminus[0]), int(Xhatminus[1])
        img, _x1, _y1, _x2, _y2 = crop_around_center(img, center_x, center_y, 192, 192)
        outputs, _ = predictor.inference(img)
        outputs = outputs[0]
        if outputs is None:
            break

        # restore the location in original image
        outputs[:, 0] += _x1
        outputs[:, 1] += _y1
        outputs[:, 2] += _x1
        outputs[:, 3] += _y1
        # measurement update
        xmin, ymin, xmax, ymax  = outputs[0, :4]
        _center_x = int((xmin + xmax) / 2)
        _center_y = int((ymin + ymax) / 2)
        measurement = [_center_x, _center_y]
        
        K = np.matmul(np.matmul(Pminus, np.transpose(Observation_Matrix)),
                             np.linalg.inv(np.matmul(np.matmul(Observation_Matrix, Pminus),
                                                     np.transpose(Observation_Matrix)) + R))
        Xhat = Xhatminus + np.matmul(K, (measurement - np.matmul(Observation_Matrix, Xhatminus)))
        P = np.matmul(np.eye(4) - np.matmul(K, Observation_Matrix), Pminus)


        result_image = predictor.visual(outputs, img_info, predictor.confthre, crop_region=(_x1, _y1, _x2, _y2))
        detection_list.append(result_image)
        if save_result:
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    
    if os.path.exists(save_folder) and detection_list:
        save_tracking_file = os.path.join(save_folder, "tracking.gif")
        create_gif(detection_list, save_tracking_file)

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, VOC_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()

    # simple_tracking(predictor, vis_folder, args.path, current_time, args.save_result)
    # print(args.center)
    kalman_tracking(predictor, vis_folder, args.path, current_time, args.save_result, args.center)
    

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
