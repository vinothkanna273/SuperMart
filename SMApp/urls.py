from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='dashboard-index'),

    path('floors', views.FloorList, name='floors'),
    path('build', views.build, name='build'),
    path('save-floor-data', views.save_floor_data, name='save-floor-data'),
    path('floor/<str:floor_name>/', views.view_floor, name='view-floor'),

    path('api/update-cart/', views.update_cart, name='update_cart'),
    
    path('customer/upload/', views.uploadItemsList, name='upload-items'),
    path('api/get-detected-items/', views.get_detected_items, name='get_detected_items'),
    path('customer/floor/<str:floor_name>/', views.customer_view_floor, name='view-customer-floor'),

]