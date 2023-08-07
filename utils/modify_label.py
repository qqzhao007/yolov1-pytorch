import os
import cv2
import xml.etree.ElementTree as ET

def modify_voc2007_label():
    # 将VOC2007数据集标签改写为yolo格式
    CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',     'aeroplane', 'bicycle', 'boat', 'bus', 'car','motorbike', 'train', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

    xml_path = './dataset/VOC2007/Annotations'
    xml_names = os.listdir(xml_path)

    for xml_name in xml_names:
        print(xml_name)
        f = open(os.path.join(xml_path, xml_name), 'r')
        tree = ET.parse(f)
        root = tree.getroot()

        # 得到图片的尺寸信息
        size = root.find('size')
        width, height = int(size.find('width').text), int(size.find('height').text)

        f2 = open('./dataset/VOC2007/labels/' + xml_name.split('.')[0] + '.txt', 'a')
        # 遍历所有的目标框
        for obj in root.iter('object'):
            c = obj.find('name').text
            difficult = obj.find('difficult').text
            if c not in CLASSES:
                continue
            box = obj.find('bndbox')
            x1, y1 = int(box.find('xmin').text), int(box.find('ymin').text)
            x2, y2 = int(box.find('xmax').text), int(box.find('ymax').text)

            x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            x, y, w, h = x / width, y / height, w / width, h / height
            print(x, y, w, h, c)

            # 将生成的标签文件保存
            f2.write('{} {} {} {} {}\n'.format(str(round(x, 8)), str(round(y, 8)), str(round(w, 8)), str(round(h, 8)), str(CLASSES.index(c))))

        f2.close()
        f.close()


if __name__ == '__main__':
    modify_voc2007_label()