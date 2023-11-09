from django.urls import path
from django.conf.urls.static import static
from . import views
from django.conf import settings

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_video, name='upload_video'),
    path('play/<int:video_id>/', views.play_video, name='play_video'),
    path('list/', views.video_list, name='video_list'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
