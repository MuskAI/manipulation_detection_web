"""ManipulationDetection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
import views as page_views
from django.views.static import serve


urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^uploadFile', page_views.upload_file),
    url(r'^$', page_views.upload_page),
    url(r'^result/(?P<path>.*)$', serve, {'document_root':'/home/liu/chenhaoran/manipulation_detection_web/upload_dir/input'}),
    url(r'^images/(?P<path>.*)$', serve, {'document_root':'/home/liu/chenhaoran/manipulation_detection_web/upload_dir/output'}),
    url(r'^im_result/(?P<path>.*)$', serve, {'document_root':'/home/liu/chenhaoran/manipulation_detection_web/upload_dir/im_output'})
]
