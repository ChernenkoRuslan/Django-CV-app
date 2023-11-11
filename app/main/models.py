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
    image = models.ImageField(upload_to='images/', blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    detected_image = models.ImageField(
        upload_to='detected_images/', blank=True)

    def __str__(self):
        return self.name

    # def save(self, *args, **kwargs):
    #     super().save(*args, **kwargs)
    #     image_path = self.image.path

    #     # Обнаружение объектов в видео и сохранение обнаруженного видео
    #     detector = ModelDetected()
    #     print('путь к текущему видео: ', image_path)
    #     print('путь к обнаруженному видео: ', "media/" +
    #           ".".join(self.image.name.split(".")[:-1]) + "_output.jpg")

    #     detector.load_photos(image_path, image_path, "media/" +
    #                          ".".join(self.video.name.split(".")[:-1]) + "_output.jpg")
    #     # detector.load_video(video_path, video_path, "output.mp4")
    #     detector.detected()
    #     print(detector.history)
    #     detected_image_path = detector.output_file
    #     print(detected_image_path)
    #     # Сохранение обнаруженного видео
    #     self.detected_image.save(detected_image_path, open(
    #         detected_image_path, "rb"), save=False)

    #     # Сохранение только определенных полей модели
    #     super().save(update_fields=['detected_image'])


# class Signal_record(models.Model):
#     name = models.CharField(max_length=100)
#     image = models.ForeignKey(Image, on_delete=models.CASCADE)
#     timestamp = models.FloatField()

#     def __str__(self):
#         return self.name
