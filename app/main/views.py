from django.shortcuts import render, redirect
from .models import Image
from .forms import ImageForm
from django.conf import settings
from django.views.generic import ListView
import random
import string
from django.core.files.storage import FileSystemStorage
# Create your views here.


def index(request):
    return render(request, 'main/index.html')


def upload_image(request):
    error = ''
    form = ImageForm(request.POST, request.FILES)
    if request.method == 'POST':
        if form.is_valid():
            image = form.save()
            return redirect('show_image', image_id=image.id)
        else:
            error = 'Форма заполнена неверно'
            return render(request, 'main/upload_image.html', {'form': form})

    return render(request, 'main/upload_image.html', {'form': form, 'error': error})


def show_image(request, image_id):
    image = Image.objects.get(id=image_id)
    person_probability = "{:.2%}".format(image.person)
    person_warning = "{:.2%}".format(image.warning)
    helmet_probability = "{:.2%}".format(image.seg_warning)
    print(person_probability)
    random_string = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=10))
    return render(request,
                  'main/play_video.html',
                  {'image': image,
                   'person_probability': person_probability,
                   'person_warning': person_warning,
                   'helmet_probability': helmet_probability,
                      'random_string': random_string,
                      'MEDIA_URL': settings.MEDIA_URL})


def image_list(request):
    images = Image.objects.all()
    return render(request, 'main/images_list.html', {'images': images})
