from django.urls import path
from django.conf.urls.static import static
from . import views
from django.conf import settings

urlpatterns = [
    path('', views.index, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
    path('show/<int:image_id>/', views.show_image, name='show_image'),
    path('list/', views.image_list, name='image_list'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
