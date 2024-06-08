"""
URL configuration for crime_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# urls.py
from django.contrib import admin
from django.urls import path, re_path
from django.views.generic import RedirectView
from myapp.views import signup_view, login_view, logout_view, video_upload, user_profile, delete_video, admin_dashboard,generate_report
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('video/upload/', video_upload, name='video_upload'),
    path('admin-dashboard/',admin_dashboard, name='admin_dashboard'),
    path('generate_report/',generate_report, name='generate_report'),
    path('user_profile/', user_profile, name='user_profile'),
    path('delete_video/<int:pk>/', delete_video, name='delete_video'),
    re_path(r'^$', RedirectView.as_view(url='login/', permanent=False)),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


