from ultralytics import YOLO
import cv2
from shapely.geometry import box, Polygon

import numpy as np
import pandas as pd

import os
import glob
import re


class ModelDetected:
    """Класс детекции нахождения персонала в опасных зонах"""

    def __init__(self,
                 model: YOLO,
                 conf_level: float = 0.3,
                 verbose: bool = False,
                 device: str = 'cpu'
                 ):
        self.scale_percent = 50
        self.conf_level = conf_level
        self.iou = 0.5
        self.verbose = verbose
        self.device = device
        self.input_dir = None
        self.classes = [0]  # 0: 'person'
        # Загружаем модель
        self.model = model
        # Словарь опасных зон, с указанием координат
        self.danger_zones = {}
        # Словарь фотографий, с указанием зон
        self.photos_by_zone = {}
        self.data_columns = ['file_name', 'count_persons',
                             'person', 'warning', 'seg_warning', 'helmet']
        self.result_df = pd.DataFrame(columns=self.data_columns)

    def load_danger_zones(self, path_zones: str):
        """Загружаем координаты опасных зон
            Заполняется словарь self.danger_zones - Словарь опасных зон, с указанием координат
        :param path_zones - путь к списку координат зон в формате *.txt
        """
        files_danger_zones = glob.glob(path_zones + "/*.txt")
        for fname in files_danger_zones:
            zone_name = fname.strip().split(
                '\\')[-1].split('/')[-1].split('.')[0]
            # Для одной камеры может быть несколько опасных зон
            zone_name = re.sub('_zone\d+', '', zone_name)
            with open(fname, 'r') as f:
                coords = f.read()
            coords = [list(map(int, re.findall(r'\d+', coord)))
                      for coord in re.findall(r'\[.+?\]', coords)]
            if zone_name not in self.danger_zones:
                self.danger_zones[zone_name] = []
            self.danger_zones[zone_name].append(np.array(coords, np.int32))

    def load_photos(self, path_cameras: str, file_types: tuple = ('*.jpg', '*.jpeg', '*.png', '*.gif')):
        """Загружаем фотографии для детекции
            Заполняется словарь self.photos_by_zone - Словарь фотографий, с указанием зон
        :param path_cameras - путь к файлам. Директория должна содержать поддиректории опасных зон
        """

        pathes_zones = [f.path for f in os.scandir(path_cameras) if f.is_dir()]
        for dir_zone in pathes_zones:
            zone_name = dir_zone.strip().split(
                '\\')[-1].split('/')[-1].split('.')[0]
            if zone_name not in self.photos_by_zone:
                self.photos_by_zone[zone_name] = []
            for filetype in file_types:
                files_photos = glob.glob(dir_zone + f"/{filetype}")
                self.photos_by_zone[zone_name].extend(files_photos)

    # Расчет пересечения (Intersection), объединения (Union) и IoU
    def intersectionOverUnion(self, pol1_xy, pol2_xy):
        # Опредяем полигоны из набора точек
        polygon1_shape = Polygon(pol1_xy)
        polygon2_shape = Polygon(pol2_xy)

        # Расчитываем пересечение (Intersection) и объединение (Union) и IOU,
        # Необходимо для расчета процента нахождения человека в опасной зоне
        polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
        polygon_union = polygon1_shape.union(polygon2_shape).area
        IoU = polygon_intersection / polygon_union
        return polygon_intersection, polygon_union, IoU

    def model_predict(self, img_files: list, zone_name: str, output_dir_zone: str):
        self.alpha = 0.2
        danger_zones = self.danger_zones[f"danger_{zone_name}"]

        for imgage_file in img_files:
            file_name = imgage_file.strip().split('\\')[-1].split('/')[-1]
            input_image = cv2.imread(imgage_file)

            result = self.model.predict(input_image, conf=self.conf_level, iou=self.iou, classes=self.classes,
                                        device=self.device, verbose=False)

            # Получаем боундбоксы и сегменты
            boxes = result[0].boxes.xyxy.cpu().numpy()
            confs = result[0].boxes.conf.cpu().numpy()
            classes = result[0].boxes.cls.cpu().numpy()

            is_segments = result[0].masks is not None
            if is_segments:
                segments = result[0].masks.xy
            for danger_zone in danger_zones:
                danger_zones_pts = danger_zone.reshape((-1, 1, 2))
                # Рисуем опасную зону
                danger_zone_image = input_image.copy()
                cv2.polylines(danger_zone_image, pts=[danger_zones_pts], isClosed=True, color=(0, 168, 255),
                              thickness=2)
                cv2.fillPoly(danger_zone_image, pts=[
                             danger_zones_pts], color=(0, 168, 255))
                input_image = cv2.addWeighted(
                    danger_zone_image, self.alpha, input_image, 1 - self.alpha, 0)
                cv2.polylines(input_image, pts=[danger_zones_pts], isClosed=True, color=(
                    0, 168, 255), thickness=2)

            image_stat = {'file_name': file_name, 'count_persons': 0, 'person': 0, 'warning': 0, 'seg_warning': 0,
                          'helmet': 0}

            print(len(boxes))
            for idx in range(len(boxes)):
                box = boxes[idx].astype('int32')
                confidence = confs[idx]
                if is_segments:
                    segment = segments[idx].astype('int32')
                detect_class = classes[idx]
                xmin, ymin, xmax, ymax = box.astype('int')
                print(box)

                person_polygon = np.array(
                    [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
                person_warning = 0
                segment_person_warning = 0
                for danger_zone in danger_zones:
                    # Расчитываем пересечение (Intersection) и объединение (Union) и IOU,
                    polygon_intersection, polygon_union, IoU = self.intersectionOverUnion(
                        person_polygon, danger_zone)
                    # Расчитываем процент нахождения человека в опасной зоне
                    person_warning = max(
                        person_warning, polygon_intersection / Polygon(person_polygon).area)
                    if is_segments:
                        # Расчитываем пересечение (Intersection) и объединение (Union) и IOU,
                        segment_polygon_intersection, _, _ = self.intersectionOverUnion(
                            segment, danger_zone)
                        # Расчитываем процент нахождения человека в опасной зоне по сегменту
                        segment_person_warning = max(segment_person_warning,
                                                     segment_polygon_intersection / Polygon(segment).area)

                #                     print(f"polygon_intersection: {polygon_intersection}")
                #                     print(f"polygon_union: {polygon_union}")
                #                     print(f"IoU: {IoU}")
                #                     print(f"person warning: {person_warning}")
                #                     print(f"person seg_warning: {segment_person_warning}")

                helmet = 0.53

                category = classes[idx].astype('int')
                center_x, center_y = int(
                    ((xmax + xmin)) / 2), int((ymax + ymin) / 2)

                if person_warning > 0.15:
                    text_color = (76, 0, 255)
                #                     elif person_warning >= 0.15:
                #                         text_color = (25, 211, 249)
                else:
                    text_color = (166, 32, 27)

                # Рисуем boundbox
                cv2.rectangle(input_image, (xmin, ymin),
                              (xmax, ymax), text_color, 2)  # box
                cv2.putText(img=input_image, text=f'Person : {int(confidence * 100)}%',
                            org=(xmin, ymin - 70), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=text_color,
                            thickness=1)
                cv2.putText(img=input_image, text=f'Warning: {int(person_warning * 100)}%',
                            org=(xmin, ymin - 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=text_color,
                            thickness=1)
                cv2.putText(img=input_image, text=f'Segment W: {int(segment_person_warning * 100)}%',
                            org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=text_color,
                            thickness=1)

                if is_segments:
                    # Рисуем сегментацию
                    segment_image = input_image.copy()
                    cv2.polylines(img=segment_image, pts=[
                                  segment], isClosed=True, color=(129, 176, 30), thickness=2)
                    cv2.fillPoly(segment_image, pts=[
                                 segment], color=(150, 190, 37))
                    input_image = cv2.addWeighted(
                        segment_image, self.alpha, input_image, 1 - self.alpha, 0)
                    cv2.polylines(img=input_image, pts=[
                                  segment], isClosed=True, color=(129, 176, 30), thickness=2)

                # Выводим итоговое изображение
                # plt.figure(figsize = (20,22))
                # plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
                # Сохрнаяем файл со статистикаой по изображению
                info_predict = [file_name, confidence, person_warning, helmet]

                image_stat['count_persons'] += 1
                image_stat['person'] = max(image_stat['person'], confidence)
                image_stat['warning'] = max(
                    image_stat['warning'], person_warning)
                image_stat['seg_warning'] = max(
                    image_stat['seg_warning'], segment_person_warning)
                image_stat['helmet'] = max(image_stat['helmet'], helmet)

            # Сохрнаяем итоговое изображение
            output_filename = output_dir_zone + "/result_" + file_name
            cv2.imwrite(output_filename, input_image)
            self.result_df.loc[len(self.result_df.index)] = image_stat
            # self.result_df.loc[len(self.result_df.index)] = info_predict

            with open(output_dir_zone + "/result_" + file_name.split(".")[0] + ".txt", 'w') as txt_file:
                txt_stat = [str(image_stat[col_name])
                            for col_name in self.data_columns]
                txt_file.write("\t".join(txt_stat))
            return output_filename

    def detected_by_dir(self, input_dir: str, file_types: str, output_dir: str):
        self.load_photos(path_cameras=input_dir, file_types=file_types)
        for zone in tqdm(self.photos_by_zone.keys()):
            output_dir_zone = f"{output_dir}/{zone}"
            # Создаем выходную директорию для зоны если её нет
            os.makedirs(output_dir_zone, exist_ok=True)
            for file_photo in tqdm(self.photos_by_zone[zone]):
                self.model_predict(
                    img_files=[file_photo], zone_name=zone, output_dir_zone=output_dir_zone)
        return self.result_df

    def detected_by_file(self, input_file, zone_name, output_dir: str):
        output_dir_zone = f"{output_dir}/{zone_name}"
        # Создаем выходную директорию для зоны если её нет
        os.makedirs(output_dir_zone, exist_ok=True)
        output_filename = self.model_predict(img_files=[input_file], zone_name=zone_name,
                                             output_dir_zone=output_dir_zone)
        return self.result_df, output_filename


if __name__ == '__main__':
    # Пути
    PATH = ''
    # Путь к данным
    DATASET_PATH = PATH + 'mini_train_dataset_train/'
    # Путь к опасным зонам
    DANGER_ZONES_PATH = DATASET_PATH + 'danger_zones/'
    # Путь к фотографиям
    CAMERAS_PATH = DATASET_PATH + 'cameras/'
    # Путь выходных данных
    OUTPUT_PATH = PATH + 'output/'

    # Создаем класс детекции
    # device = 'cuda'
    device = 'cpu'
    # Загружаем модель
    model_seg = YOLO("yolov8x-seg.pt")

    detector = ModelDetected(model=model_seg, device=device)
    # Загрузка опасных зон
    detector.load_danger_zones(path_zones=DANGER_ZONES_PATH)

    # Анализ фотографий из директорий
    # file_types = ('*.jpg', '*.jpeg', '*.png', '*.gif')
    # result_df = detector.detected_by_dir(input_dir=CAMERAS_PATH, file_types=file_types, output_dir = OUTPUT_PATH)

    # Анализ одного файла с фотографией
    test_zone = 'Spp-K1-1-2-6'
    test_file = CAMERAS_PATH + \
        f'{test_zone}/ce42257e-1199-46e7-90f7-78499facc320.jpg'
    result_df, output_filename = detector.detected_by_file(
        input_file=test_file, zone_name=test_zone, output_dir=OUTPUT_PATH)
    result_dict = result_df.to_dict('records')
    print(result_dict)
    print(output_filename)
