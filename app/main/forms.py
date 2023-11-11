from django.forms import ModelForm
from django import forms
from django.db import models
from .models import Image


class ImageForm(ModelForm):
    class Meta:
        model = Image
        fields = ['name', 'record_date', 'workshop', 'description', 'image']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Введите название', 'label': 'Название записи'}),
            'record_date': forms.DateInput(attrs={'placeholder': 'выберите  дату', 'label': 'Дата снимка', 'type': 'date'}),
            'workshop': forms.Select(attrs={'placeholder': 'Выберите площадку', 'label': 'Площадка', }),
            'description': forms.Textarea(attrs={'placeholder': 'Введите описание', 'label': 'Описание'}),
        }

    def __init__(self, *args, **kwargs):
        super(ImageForm, self).__init__(*args, **kwargs)
        self.fields['name'].label = 'Название записи'
        self.fields['record_date'].label = 'Дата снимка'
        self.fields['workshop'].label = 'Площадка'
        self.fields['description'].label = 'Описание'

# class VideoForm(forms.ModelForm):
#     Name = models.CharField(max_length=100)
#     Record_date = models.DateField()
#     Workshop = models.CharField(max_length=100)
#     Description = models.TextField()
#     Video = models.FileField(upload_to='videos/')
#     Upload_date = models.DateTimeField(auto_now_add=True)
