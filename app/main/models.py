from collections.abc import Iterable
from django.db import models
from .basemodel import ModelDetected
from ultralytics import YOLO
import os

# Create your models here.

WORKSHOP_CHOICES = [
    ('DpR-Csp-uipv-ShV-V1', 'DpR-Csp-uipv-ShV-V1'),
    ('Pgp-com2-K-1-0-9-36', 'Pgp-com2-K-1-0-9-36'),
    ('Pgp-lpc2-K-0-1-38', 'Pgp-lpc2-K-0-1-38'),
    ('Phl-com3-Shv2-9-K34', 'Phl-com3-Shv2-9-K34'),
    ('Php-Angc-K3-1', 'Php-Angc-K3-1'),
    ('Php-Angc-K3-8', 'Php-Angc-K3-8'),
    ('Php-Ctm-K-1-12-56', 'Php-Ctm-K-1-12-56'),
    ('Php-Ctm-Shv1-2-K3', 'Php-Ctm-Shv1-2-K3'),
    ('Php-nta4-shv016309-k2-1-7', 'Php-nta4-shv016309-k2-1-7'),
    ('Spp-210-K1-3-3-5', 'Spp-210-K1-3-3-5'),
    ('Spp-210-K1-3-3-6', 'Spp-210-K1-3-3-6'),
    ('Spp-K1-1-2-6_zone1', 'Spp-K1-1-2-6_zone1'),
    ('Spp-K1-1-2-6_zone2', 'Spp-K1-1-2-6_zone2'),
    ('Spp-K1-1-2-6_zone3', 'Spp-K1-1-2-6_zone3'),
    ('Spp-K1-1-2-6_zone4', 'Spp-K1-1-2-6_zone4'),
]


class Image(models.Model):
    name = models.CharField(max_length=100)
    record_date = models.DateField()
    workshop = models.CharField(max_length=100, choices=WORKSHOP_CHOICES)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='images/%Y/%m/%d/', blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    detected_image = models.ImageField(
        upload_to='detected_images/', blank=True)
    count_persons = models.IntegerField(default=0)
    person = models.FloatField(default=0)
    warning = models.FloatField(default=0)
    seg_warning = models.FloatField(default=0)
    # result = [{'file_name': '08e4ecce-be75-4064-aff2-ce6b18b56934_itqB9qP.jpg', 'count_persons': 1,
    #            'person': 0.6837488412857056, 'warning': 0, 'seg_warning': 0, 'helmet': 0.53}]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        image_path = self.image.path
        zone_name = self.workshop.replace('danger_', '')
        print(zone_name)

        # Обнаружение объектов в видео и сохранение обнаруженного видео
        device = 'cpu'
        # Загружаем модель
        model_seg = YOLO("yolov8x-seg.pt")
        detector = ModelDetected(model=model_seg, device=device)
        print(os.getcwd())
        print(os.listdir("main/danger_zones"))

        # Загрузка опасных зон
        detector.load_danger_zones(path_zones="main/danger_zones")
        print(detector.danger_zones.keys())
        test_zone = 'DpR-Csp-uipv-ShV-V1'
        test_file = image_path
        result_df, output_filename = detector.detected_by_file(
            input_file=test_file, zone_name=test_zone, output_dir='media/output')
        result_dict = result_df.to_dict('records')
        print(result_dict)
        print(output_filename)
        self.detected_image = output_filename.replace(
            'media/', '')
        self.count_persons = result_dict[0]['count_persons']
        self.person = result_dict[0]['person']
        self.warning = result_dict[0]['warning']
        self.seg_warning = result_dict[0]['seg_warning']

        # Сохранение только определенных полей модели
        super().save(update_fields=[
            'detected_image', 'count_persons', 'person', 'warning', 'seg_warning'])
