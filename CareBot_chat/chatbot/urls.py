from django.urls import path
from . import views
from django.contrib import admin

urlpatterns = [
    # path('admin/', admin.site.urls),  # Admin route
    path('chat/', views.chat_view, name='chat'),  # Route for the chat page

]
