from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # path('', views.index, name='dashboard-index'),
    path('', views.analytics_dashboard, name='dashboard-index'),

    path('history', views.history, name='history'),

    path('floors', views.FloorList, name='floors'),
    path('build', views.build, name='build'),
    path('save-floor-data', views.save_floor_data, name='save-floor-data'),
    path('floor/<str:floor_name>/', views.view_floor, name='view-floor'),

    path('items/', views.ItemsList, name='items-list'),
    path('items/add/', views.add_item, name='add-item'),
    path('items/edit/<int:pk>/', views.edit_item, name='edit-item'),
    path('items/delete/<int:pk>/', views.delete_item, name='delete-item'),
    path('search-item', views.search_item, name='search-item'),
    
    path('api/update-cart/', views.update_cart, name='update_cart'),
    
    path('customer/upload/', views.uploadItemsList, name='upload-items'),
    path('api/get-detected-items/', views.get_detected_items, name='get_detected_items'),
    path('customer/floor/<str:floor_name>/', views.customer_view_floor, name='view-customer-floor'),

    path('payment-confirmation/', views.payment_confirmation, name='payment_confirmation'),
    path('process-payment/', views.process_payment, name='process_payment'),
    path('payment-success/', views.payment_success, name='payment_success'),

    path('analytics/item/<int:item_id>/', views.item_analytics_view, name='item_analytics'),
    path('debug-forecast/', views.debug_forecast_data, name='debug-forecast'),

    # Enhanced Analytics URLs
    path('enhanced/', views.analytics_dashboard_enhanced, name='enhanced-dashboard'),
    path('item/<int:item_id>/analytics/enhanced/', views.item_analytics_view_enhanced, name='enhanced-item-analytics'),
    path('api/category-comparison/', views.get_category_comparison_report, name='get-category-comparison-report'),

    # Additional enhanced endpoints
    path('api/advanced-metrics/<int:item_id>/', views.get_item_advanced_metrics, name='get-item-advanced-metrics'),


]