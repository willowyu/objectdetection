from xml.etree import ElementTree as ET
import os
import xml.etree.ElementTree as ET
import pickle



def convert_voc_to_yolo(voc_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for xml_file in os.listdir(os.path.join(voc_dir, "Annotations")):
        tree = ET.parse(os.path.join(voc_dir, "Annotations", xml_file))
        root = tree.getroot()
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        txt_file = xml_file.replace(".xml", ".txt")
        with open(os.path.join(output_dir, txt_file), "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                class_id = voc_classes.index(class_name)  # 需提前定义 VOC 类别列表
                bbox = obj.find("bndbox")
                x_min = float(bbox.find("xmin").text)
                y_min = float(bbox.find("ymin").text)
                x_max = float(bbox.find("xmax").text)
                y_max = float(bbox.find("ymax").text)

                # 转换为 YOLO 格式
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# VOC 类别列表（20类）
voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]




def process_datasets():
    sets = ['train', 'test', 'val']

    Imgpath = 'E:/ObjectDetection/yolov5/datasets/VOC2007/images' 
    ImageSets_path = './datasets/VOC2007/ImageSets/' #合并
    Label_path = './datasets/VOC2007/'

    convert_voc_to_yolo(Label_path,os.path.join(Label_path, "labels"))

    for image_set in sets:
        # if not os.path.exists(Label_path + 'labels/'):
        #     os.makedirs(Label_path + 'labels/')
        image_ids = open(ImageSets_path + '%s.txt' % (image_set)).read().strip().split()
        list_file = open(Label_path + '%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write(Imgpath + '/%s.jpg\n' % (image_id))
            #convert_voc_to_yolo(Label_path,os.path.join(Label_path, "labels"))
        list_file.close()

if __name__ == '__main__':
    process_datasets()