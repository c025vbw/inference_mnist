# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('cnn/', views.predict, name='predict'),
]
