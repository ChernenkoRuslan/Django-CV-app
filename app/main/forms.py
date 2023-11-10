from django.forms import ModelForm
from django import forms
from django.db import models
from .models import Video


class VideoForm(ModelForm):
    class Meta:
        model = Video
        fields = ['name', 'record_date', 'workshop', 'description', 'video']

    def __init__(self, *args, **kwargs):
        super(VideoForm, self).__init__(*args, **kwargs)
        self.fields['video'].required = True


# class VideoForm(forms.ModelForm):
#     Name = models.CharField(max_length=100)
#     Record_date = models.DateField()
#     Workshop = models.CharField(max_length=100)
#     Description = models.TextField()
#     Video = models.FileField(upload_to='videos/')
#     Upload_date = models.DateTimeField(auto_now_add=True)
