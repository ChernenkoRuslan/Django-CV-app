from django.shortcuts import render, redirect
from .models import Video
from .forms import VideoForm
from django.conf import settings
from django.views.generic import ListView
import random
import string

# Create your views here.


def index(request):
    return render(request, 'main/index.html')


def upload_video(request):
    error = ''
    form = VideoForm()
    video = Video()
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('play_video', video_id=video.id)
        else:
            error = 'Форма заполнена неверно'
            return render(request, 'main/upload_video.html', {'form': form})
    # if request.method == 'POST':
    #     title = request.POST['title']
    #     file = request.FILES['file']

    #     video = Video()
    #     video.name = title
    #     video.video = file
    #     video.save()

    #     return redirect('play_video', video_id=video.id)

    return render(request, 'main/upload_video.html', {'form': form, 'error': error})


def play_video(request, video_id):
    video = Video.objects.get(id=video_id)
    random_string = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=10))
    return render(request,
                  'main/play_video.html',
                  {'video': video,
                      'random_string': random_string,
                      'MEDIA_URL': settings.MEDIA_URL})


def video_list(request):
    videos = Video.objects.all()
    return render(request, 'main/video_list.html', {'videos': videos})

# class VideoListView(ListView):
#     model = Video

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['MEDIA_URL'] = settings.MEDIA_URL
#         return context
