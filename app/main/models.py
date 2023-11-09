from collections.abc import Iterable
from django.db import models
from .cvmodel import ModelDetected
from ultralytics import YOLO
import os

# Create your models here.


class Video(models.Model):
    name = models.CharField(max_length=100)
    video = models.FileField(upload_to='videos/')
    upload_date = models.DateTimeField(auto_now_add=True)
    screenshot = models.ImageField(
        upload_to='screenshots/', default='static/no_image.jpg', blank=True)
    detected_video = models.FileField(upload_to='detected_videos/', blank=True)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        video_path = self.video.path

        # Обнаружение объектов в видео и сохранение обнаруженного видео
        dict_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 16: 'dog', 17: 'horse', 18: 'sheep',
                        19: 'cow', 21: 'bear', }
        model = YOLO('yolov8n.pt')
        detector = ModelDetected(model=model, type_cam='front', skip_frames=5,
                                 conf_level=0.7, verbose=True, dict_classes=dict_classes, device='cpu')

        print('путь к текущему видео: ', video_path)
        print('путь к обнаруженному видео: ', "media/" +
              ".".join(self.video.name.split(".")[:-1]) + "_output.mp4")

        detector.load_video(video_path, video_path, "media/" +
                            ".".join(self.video.name.split(".")[:-1]) + "_output.mp4")
        # detector.load_video(video_path, video_path, "output.mp4")
        detector.detected()
        print(detector.history)
        detected_video_path = detector.output_file
        print(detected_video_path)
        # Сохранение обнаруженного видео
        self.detected_video.save(detected_video_path, open(
            detected_video_path, "rb"), save=False)

        # Сохранение только определенных полей модели
        super().save(update_fields=['detected_video'])


class Signal_record(models.Model):
    name = models.CharField(max_length=100)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    timestamp = models.FloatField()

    def __str__(self):
        return self.name
