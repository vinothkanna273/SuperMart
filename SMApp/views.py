from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse
from django.core.cache import cache
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Sum
from django.db import models
from django.utils import timezone

import json
import csv
import docx
import urllib.parse
from collections import deque
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from decimal import Decimal
import datetime
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .forms import ItemForm
from .models import Item, Floor, Shelf, Transaction

import logging
logger = logging.getLogger(__name__)


# dashboard
def index(request):
    # pull data from db, transform, send email
    # return HttpResponse('Hello World')
    return render(request, 'AApp/index.html')

# history
def history(request):
    transactions_list = Transaction.objects.all().order_by('-timestamp')
    # no_of_items = 15
    # paginator = Paginator(transactions_list, no_of_items)
    # page = request.GET.get('page')
    # try:
    #     items = paginator.page(page)
    # except PageNotAnInteger:
    #     items = paginator.page(1)
    # except EmptyPage:
    #     items = paginator.page(paginator.num_pages)

    context = {
        'items': transactions_list,
    }
    return render(request, 'AApp/history.html', context)



# items
def ItemsList(request):
    items_list = Item.objects.all()
    no_of_items = 10
    paginator = Paginator(items_list, no_of_items)
    page = request.GET.get('page')
    try:
        items = paginator.page(page)
    except PageNotAnInteger:
        items = paginator.page(1)
    except EmptyPage:
        items = paginator.page(paginator.num_pages)

    context = {
        'items': items
    }
    return render(request, 'AApp/Item/items.html', context)

def add_item(request):
    if request.method == 'POST':
        form = ItemForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Item added successfully.')
            return redirect('items-list')  
    else:
        form = ItemForm()
    return render(request, 'AApp/Item/add_Items.html', {'form': form})

def edit_item(request, pk):
    item = get_object_or_404(Item, pk=pk)
    if request.method == 'POST':
        form = ItemForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
            messages.success(request, 'Item edited successfully.')
            return redirect('items-list')
    else:
        form = ItemForm(instance=item)
    return render(request, 'AApp/Item/edit_Item.html', {'form': form})

def delete_item(request, pk):
    item = get_object_or_404(Item, pk=pk)
    if request.method == 'POST':
        item.delete()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'status': 'success'}, status=200)
        else:
            messages.success(request, 'Item deleted successfully.')
            return redirect('items-list')
    return redirect('items-list')

@csrf_exempt
def search_item(request):
    if request.method == 'POST':
        search_str = json.loads(request.body).get('searchText', '')
        items = Item.objects.filter(name__icontains=search_str) | \
                   Item.objects.filter(description__icontains=search_str) | \
                   Item.objects.filter(price__icontains=search_str)
        data = list(items.values())
        return JsonResponse(data, safe=False)



# floor
def FloorList(request):
    floors = Floor.objects.all()
    return render(request, 'AApp/build/floors.html', {'floors': floors})

def build(request):
    items = Item.objects.all()
    context = {
        'items': items
    }
    return render(request, 'AApp/build/build.html', context)

@csrf_exempt
def save_floor_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name')
            length = data.get('length')
            width = data.get('width')
            shelves = data.get('shelves')

            try:
                floor = Floor.objects.get(name=name)
                floor.length = length
                floor.width = width
                floor.data = shelves
                floor.save()
            except ObjectDoesNotExist:
                floor = Floor.objects.create(
                    name=name, length=length, width=width, data=shelves)

            Shelf.objects.filter(floor=floor).delete()

            for cell_index, shelf_data in shelves.items():
                item_ids = shelf_data.get('itemIds', [])
                item_names = shelf_data.get('itemNames', [])
                mode = shelf_data.get('mode', None)

                if mode is not None:
                    Shelf.objects.create(
                        floor=floor, cell_index=cell_index, item_ids=item_ids, item_names=item_names, mode=mode)

            return JsonResponse({'message': 'Floor data saved successfully'})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            logger.error(f"Error saving floor data: {e}", exc_info=True)
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def view_floor(request, floor_name):
    try:
        items = Item.objects.all()
        floor = Floor.objects.get(name=floor_name)
        shelves = Shelf.objects.filter(floor=floor).values(
            'cell_index', 'mode', 'item_ids', 'item_names')
        
        # Convert shelves to a dictionary format that matches your grid data
        shelves_dict = {shelf['cell_index']: {
            'mode': shelf['mode'],
            'itemIds': shelf['item_ids'],
            'itemNames': shelf['item_names']
        } for shelf in shelves}
        
        context = {
            'items': items,
            'floor': floor, 
            'shelves': list(shelves),
            'floor_data_json': json.dumps(floor.data)  # Add this line
        }
        return render(request, 'AApp/build/viewFloor.html', context)
    except Floor.DoesNotExist:
        return render(request, 'AApp/build/viewFloor.html', {'floor_name': floor_name})



# customer list handling
def handle_uploaded_file(file):
    file_type = file.content_type

    try:
        if file_type in ["text/plain"]:  # Handle .txt files
            file.seek(0)
            content = file.read().decode(errors="ignore").strip()
            return content.splitlines() if content else []

        elif file_type in ["text/csv", "application/vnd.ms-excel"]:  # Handle .csv files
            file.seek(0)
            decoded_file = file.read().decode("utf-8").splitlines()
            reader = csv.reader(decoded_file)
            return [row[0] for row in reader if row]  # Extracting first column (item names)

        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel.sheet.macroEnabled.12"]:
            file.seek(0)
            df = pd.read_excel(file, usecols=[0])  # Assuming first column has item names
            return df.iloc[:, 0].dropna().tolist()

        elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            file.seek(0)
            doc = docx.Document(file)
            content = "\n".join([para.text for para in doc.paragraphs]).strip()
            return content.splitlines() if content else []

        else:
            return None

    except Exception as e:
        print("Error processing file:", str(e))
        return None

def uploadItemsList(request):
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        file_data = handle_uploaded_file(uploaded_file)

        if file_data is None:
            return HttpResponse("Invalid file type or error processing file.", status=400)

        print("\nExtracted Items from File:", file_data)

        matched_items = []
        item_names = []
        for item_name in file_data:
            item_name = item_name.strip()
            if not item_name:  # Skip empty names
                continue
                
            item_names.append(item_name)
            found_items = Item.objects.filter(name__icontains=item_name)
            if found_items.exists():
                matched_items.extend(found_items)

        if matched_items:
            print("\nMatched Items in Database:")
            for item in matched_items:
                print(f"Name: {item.name}, Quantity: {item.quantity}, Price: ₹{item.price}, Description: {item.description}")
        else:
            print("\nNo matching items found in database.")
        
        # Store the list of item names in session
        request.session['uploaded_items'] = item_names
        
        # Use the correct URL pattern for redirection
        # Using reverse() to ensure correct URL generation
        redirect_url = reverse('view-customer-floor', kwargs={'floor_name': 'ground'})
        
        # Add query parameter containing the items
        encoded_items = urllib.parse.quote(json.dumps(item_names))
        return redirect(f'{redirect_url}?uploaded_items={encoded_items}')

    return render(request, 'AApp/customer/uploadList.html')

@csrf_exempt
def customer_view_floor(request, floor_name):
    try:
        items = Item.objects.all()
        floor = Floor.objects.get(name=floor_name)
        shelves = Shelf.objects.filter(floor=floor).values(
            'cell_index', 'mode', 'item_ids', 'item_names')

        uploaded_items = request.session.get('uploaded_items', [])

        context = {
            'items': items,
            'floor': floor,
            'shelves': list(shelves),
            'floor_data_json': json.dumps(floor.data),
            'uploaded_items_json': json.dumps(uploaded_items),
            'uploaded_items': uploaded_items,
            'enable_realtime': True  # Add this flag
        }
        return render(request, 'AApp/customer/viewCustomerFloor.html', context)
    
    except Floor.DoesNotExist:
        return render(request, 'AApp/customer/viewCustomerFloor.html', {'floor_name': floor_name})

@csrf_exempt
def get_detected_items(request):
    if request.method == 'GET':
        try:
            detected_items = cache.get('detected_items', [])
            matched_items = cache.get('detected_matched_items', [])
            
            print(f"Retrieved from cache - detected_items: {detected_items}")
            print(f"Retrieved from cache - matched items count: {len(matched_items) if matched_items else 0}")
            
            return JsonResponse({
                'status': 'success',
                'detected_items': detected_items,
                'matched_items': matched_items
            })
        except Exception as e:
            print(f"Error in get_detected_items: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Only GET requests are allowed'
    }, status=405)

@csrf_exempt
def update_cart(request):
    # socket code
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            cart_id = data.get('cart_id', 'default')
            items = data.get('items', {})  
            total_cost = data.get('total_cost', 0.0)
            total_weight = data.get('total_weight', 0.0)

            matched_items = []
            detected_items = list(items.keys())  
            
            for item_name, item_details in items.items():
                quantity = item_details.get("quantity", 1)  
                
                db_items = Item.objects.filter(name__iexact=item_name)
                
                if not db_items.exists():
                    modified_name = item_name[1:]
                    db_items = Item.objects.filter(name__icontains=modified_name)
                
                if db_items.exists():
                    for db_item in db_items:
                        matched_items.append({
                            "name": db_item.name,
                            "quantity": quantity,  
                            "price": float(db_item.price),  
                            "description": db_item.description
                        })
                else:
                    print(f"Item '{item_name}' (or '{modified_name}') not found in database")

            # Store detected items in cache
            cache.set('detected_items', detected_items, timeout=None)  
            cache.set('detected_matched_items', matched_items, timeout=None)
            
            # Send update to WebSocket channel
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                'cart_updates',
                {
                    'type': 'cart_update',
                    'detected_items': detected_items,
                    'matched_items': matched_items
                }
            )

            return JsonResponse({
                'status': 'success',
                'message': 'Cart updated successfully',
                'matched_items': matched_items  
            })
        except Exception as e:
            print(f"Error in update_cart: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Only POST requests are allowed'
    }, status=405)



# payment
def payment_confirmation(request):
    # Get items from cache that were detected
    matched_items = cache.get('detected_matched_items', [])
    
    # Calculate totals for each item and overall total
    total_amount = Decimal('0.00')
    for item in matched_items:
        item['total'] = Decimal(str(item['price'])) * item['quantity']
        total_amount += item['total']
    
    context = {
        'matched_items': matched_items,
        'total_amount': total_amount,
    }
    
    return render(request, 'AApp/customer/paymentPage.html', context)

@csrf_exempt
def process_payment(request):
    if request.method == 'POST':
        try:
            # Get items from cache
            matched_items = cache.get('detected_matched_items', [])
            
            if not matched_items:
                messages.error(request, "Your cart is empty.")
                return redirect('payment_confirmation')
            
            # Create transaction records for each item
            for item_data in matched_items:
                # Get the item from database
                try:
                    item = Item.objects.get(name=item_data['name'])
                    
                    # Create transaction record
                    total_price = Decimal(str(item_data['price'])) * item_data['quantity']
                    
                    Transaction.objects.create(
                        item=item,
                        item_name=item.name,
                        quantity=item_data['quantity'],
                        total_price=total_price,
                        user=request.user if request.user.is_authenticated else None
                    )
                    
                    # Update item quantity in inventory (optional)
                    item.quantity = max(0, item.quantity - item_data['quantity'])
                    item.save()
                    
                except Item.DoesNotExist:
                    # Handle case where item no longer exists
                    messages.warning(request, f"Item '{item_data['name']}' is no longer available.")
            
            # Clear the cache after successful payment
            cache.delete('detected_items')
            cache.delete('detected_matched_items')
            
            # Send update to WebSocket to clear the cart display
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                'cart_updates',
                {
                    'type': 'cart_update',
                    'detected_items': [],
                    'matched_items': []
                }
            )
            
            messages.success(request, "Payment successful! Your transaction has been recorded.")
            return redirect('payment_success')  # Redirect to a success page
            
        except Exception as e:
            messages.error(request, f"Payment failed: {str(e)}")
            return redirect('payment_confirmation')
    
    return redirect('payment_confirmation')

def payment_success(request):
    return render(request, 'AApp/customer/payment_success.html')



# Analytics

def calculate_item_performance_metrics(item_id=None, period='monthly', months=6):
    end_date = timezone.now().date()
    start_date = end_date - datetime.timedelta(days=30*months)
    
    # Query base
    queryset = Transaction.objects.filter(timestamp__gte=start_date)
    if item_id:
        queryset = queryset.filter(item_id=item_id)
        try:
            item = Item.objects.get(id=item_id)
            current_stock = item.quantity
        except Item.DoesNotExist:
            current_stock = 0
    else:
        # For overall metrics, sum up all inventory
        current_stock = Item.objects.aggregate(Sum('quantity'))['quantity__sum'] or 0
    
    # Total sales in the period
    total_sales = queryset.aggregate(Sum('quantity'))['quantity__sum'] or 0
    
    # Calculate days in the period
    days_in_period = (end_date - start_date).days
    if days_in_period == 0:  # Avoid division by zero
        days_in_period = 1
    
    # Calculate metrics
    sales_velocity_daily = total_sales / days_in_period
    sales_velocity_weekly = sales_velocity_daily * 7
    sales_velocity_monthly = sales_velocity_daily * 30
    
    # Days of supply (how long current stock will last based on average daily sales)
    days_of_supply = None
    if sales_velocity_daily > 0:
        days_of_supply = current_stock / sales_velocity_daily
    
    # Inventory turnover (annualized)
    inventory_turnover = None
    if current_stock > 0:
        # Annualized sales / current inventory
        inventory_turnover = (total_sales / days_in_period * 365) / current_stock
    
    return {
        'sales_velocity_daily': sales_velocity_daily,
        'sales_velocity_weekly': sales_velocity_weekly,
        'sales_velocity_monthly': sales_velocity_monthly,
        'days_of_supply': days_of_supply,
        'inventory_turnover': inventory_turnover,
        'total_sales_period': total_sales,
        'current_stock': current_stock
    }

# ----------

def analytics_dashboard(request):
    """
    Improved analytics dashboard with better error handling and data validation
    """
    # Get all items
    items = Item.objects.all()
    
    # Get overall sales trend with better error handling
    try:
        overall_df = prepare_time_series_data(period='weekly')
        print(f"Overall data shape: {overall_df.shape}")
        print(f"Overall data sample:\n{overall_df.head()}")
        
        if len(overall_df) >= 3:
            overall_forecast, overall_chart, overall_metrics = forecast_with_prophet(
                overall_df, periods=12, frequency='W'
            )
        else:
            print("Insufficient overall data for forecasting")
            overall_forecast = pd.DataFrame()
            overall_chart = None
            overall_metrics = None
    except Exception as e:
        print(f"Error in overall forecasting: {str(e)}")
        overall_forecast = pd.DataFrame()
        overall_chart = None
        overall_metrics = None
    
    # Calculate overall performance metrics
    try:
        overall_performance = calculate_item_performance_metrics(period='monthly')
    except Exception as e:
        print(f"Error calculating overall performance: {str(e)}")
        overall_performance = {
            'sales_velocity_daily': 0,
            'sales_velocity_weekly': 0,
            'sales_velocity_monthly': 0,
            'days_of_supply': None,
            'inventory_turnover': None,
            'total_sales_period': 0,
            'current_stock': 0
        }
    
    # Get latest transactions for the dashboard
    latest_transactions = Transaction.objects.all().order_by('-timestamp')[:10]
    
    # Initialize lists for stockout predictions and performance metrics
    items_with_stockout = []
    items_performance = []
    
    # Process each item with better error handling
    for item in items:
        try:
            # Get item performance metrics 
            item_metrics = calculate_item_performance_metrics(item_id=item.id)
            
            # Add performance metrics to list
            items_performance.append({
                'item': item,
                'metrics': item_metrics
            })
            
            # Get forecasting data for stockout prediction
            item_df = prepare_time_series_data(item_id=item.id, period='weekly')
            
            if len(item_df) >= 3:  # Only forecast if we have sufficient data
                forecast, _, _ = forecast_with_prophet(item_df, periods=12, frequency='W')
                stockout_date = predict_stockout(item.id, forecast)
                
                if stockout_date:
                    # Calculate days until stockout
                    today = timezone.now().date()
                    if hasattr(stockout_date, 'date'):
                        days_until = (stockout_date.date() - today).days
                    else:
                        days_until = (stockout_date - today).days if isinstance(stockout_date, datetime.date) else None
                    
                    items_with_stockout.append({
                        'item': item,
                        'stockout_date': stockout_date,
                        'days_until': days_until
                    })
                    
        except Exception as e:
            print(f"Error processing item {item.id} ({item.name}): {str(e)}")
            # Add item with default metrics
            items_performance.append({
                'item': item,
                'metrics': {
                    'sales_velocity_daily': 0,
                    'sales_velocity_weekly': 0,
                    'sales_velocity_monthly': 0,
                    'days_of_supply': None,
                    'inventory_turnover': None,
                    'total_sales_period': 0,
                    'current_stock': item.quantity
                }
            })
    
    # Sort by stockout date (soonest first)
    if items_with_stockout:
        items_with_stockout.sort(key=lambda x: x['stockout_date'] if x['stockout_date'] else datetime.date.max)
    
    # Sort items by inventory turnover (highest first)
    items_performance.sort(
        key=lambda x: x['metrics']['inventory_turnover'] if x['metrics']['inventory_turnover'] is not None else 0, 
        reverse=True
    )
    
    # Prepare context with all data
    context = {
        'overall_chart': overall_chart,
        'overall_metrics': overall_metrics,
        'overall_performance': overall_performance,
        'items_with_stockout': items_with_stockout[:10],  # Limit to top 10
        'items_performance': items_performance[:20],      # Limit to top 20
        'latest_transactions': latest_transactions,
        'items': items,
        'total_items': items.count(),
        'low_stock_items': items.filter(quantity__lt=10).count(),  # Items with less than 10 units
    }
    
    return render(request, 'AApp/index.html', context)

def item_analytics_view(request, item_id):
    """
    Improved item analytics view with better error handling
    """
    try:
        # Get the item
        item = Item.objects.get(id=item_id)
        
        # Prepare time series data
        weekly_df = prepare_time_series_data(item_id=item_id, period='weekly')
        monthly_df = prepare_time_series_data(item_id=item_id, period='monthly')
        
        print(f"Item {item.name} - Weekly data shape: {weekly_df.shape}")
        print(f"Item {item.name} - Monthly data shape: {monthly_df.shape}")
        
        # Generate forecasts with error handling
        weekly_forecast = pd.DataFrame()
        weekly_chart = None
        weekly_metrics = None
        
        if len(weekly_df) >= 3:
            try:
                weekly_forecast, weekly_chart, weekly_metrics = forecast_with_prophet(
                    weekly_df, periods=12, frequency='W'
                )
            except Exception as e:
                print(f"Error in weekly forecast for item {item.name}: {str(e)}")
        
        monthly_forecast = pd.DataFrame()
        monthly_chart = None
        monthly_metrics = None
        
        if len(monthly_df) >= 3:
            try:
                monthly_forecast, monthly_chart, monthly_metrics = forecast_with_prophet(
                    monthly_df, periods=6, frequency='M'
                )
            except Exception as e:
                print(f"Error in monthly forecast for item {item.name}: {str(e)}")
        
        # Calculate performance metrics
        try:
            performance_metrics = calculate_item_performance_metrics(item_id=item_id)
        except Exception as e:
            print(f"Error calculating performance metrics for item {item.name}: {str(e)}")
            performance_metrics = {
                'sales_velocity_daily': 0,
                'sales_velocity_weekly': 0,
                'sales_velocity_monthly': 0,
                'days_of_supply': None,
                'inventory_turnover': None,
                'total_sales_period': 0,
                'current_stock': item.quantity
            }
        
        # Predict stockout date
        stockout_date = None
        if not weekly_forecast.empty:
            try:
                stockout_date = predict_stockout(item_id, weekly_forecast)
            except Exception as e:
                print(f"Error predicting stockout for item {item.name}: {str(e)}")
        
        context = {
            'item': item,
            'weekly_chart': weekly_chart,
            'monthly_chart': monthly_chart,
            'stockout_date': stockout_date,
            'weekly_metrics': weekly_metrics,
            'monthly_metrics': monthly_metrics,
            'performance_metrics': performance_metrics,
            'weekly_data_points': len(weekly_df),
            'monthly_data_points': len(monthly_df),
        }
        
        return render(request, 'AApp/analytics/item_analytics.html', context)
        
    except Item.DoesNotExist:
        messages.error(request, f'Item with ID {item_id} does not exist.')
        return redirect('items-list')
    except Exception as e:
        messages.error(request, f'Error loading analytics: {str(e)}')
        return redirect('items-list')

def debug_forecast_data(request):
    """
    Debug view to check the quality of your forecasting data
    """
    if not request.user.is_superuser:
        return HttpResponse("Access denied", status=403)
    
    debug_info = []
    
    # Check overall data
    overall_df = prepare_time_series_data(period='weekly')
    debug_info.append(f"Overall weekly data: {len(overall_df)} points")
    if not overall_df.empty:
        debug_info.append(f"Date range: {overall_df['ds'].min()} to {overall_df['ds'].max()}")
        debug_info.append(f"Value range: {overall_df['y'].min()} to {overall_df['y'].max()}")
        debug_info.append(f"Mean: {overall_df['y'].mean():.2f}, Std: {overall_df['y'].std():.2f}")
    
    debug_info.append("\n" + "="*50 + "\n")
    
    # Check individual items
    items = Item.objects.all()[:5]  # Check first 5 items
    for item in items:
        item_df = prepare_time_series_data(item_id=item.id, period='weekly')
        debug_info.append(f"Item '{item.name}': {len(item_df)} data points")
        if not item_df.empty:
            debug_info.append(f"  Date range: {item_df['ds'].min()} to {item_df['ds'].max()}")
            debug_info.append(f"  Value range: {item_df['y'].min()} to {item_df['y'].max()}")
            debug_info.append(f"  Total sales: {item_df['y'].sum()}")
    
    # Check recent transactions
    recent_transactions = Transaction.objects.all().order_by('-timestamp')[:10]
    debug_info.append(f"\nRecent transactions: {len(recent_transactions)}")
    for txn in recent_transactions:
        debug_info.append(f"  {txn.timestamp.strftime('%Y-%m-%d')}: {txn.item_name} x{txn.quantity}")
    
    return HttpResponse("<pre>" + "\n".join(debug_info) + "</pre>")

def clean_time_series_data(df):
    """
    Clean time series data by removing outliers and ensuring data quality
    """
    if df.empty or len(df) < 2:
        return df
    
    # Remove negative values
    df = df[df['y'] >= 0].copy()
    
    # Remove extreme outliers (values > 3 standard deviations from mean)
    if len(df) > 3:
        mean_val = df['y'].mean()
        std_val = df['y'].std()
        if std_val > 0:  # Avoid division by zero
            df = df[abs(df['y'] - mean_val) <= 3 * std_val].copy()
    
    # Ensure we still have enough data points
    if len(df) < 2:
        return pd.DataFrame(columns=['ds', 'y'])
    
    return df

def calculate_forecast_accuracy(actual_df, forecast_df):
    """
    Calculate various accuracy metrics for the forecast
    """
    if actual_df.empty or forecast_df.empty:
        return {'mape': None, 'rmse': None, 'mae': None}
    
    # Merge actual and forecasted data
    merged_df = pd.merge(
        actual_df, 
        forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='inner'
    )
    
    if merged_df.empty:
        return {'mape': None, 'rmse': None, 'mae': None}
    
    # Calculate errors
    merged_df['error'] = merged_df['y'] - merged_df['yhat']
    merged_df['abs_error'] = abs(merged_df['error'])
    merged_df['squared_error'] = merged_df['error'] ** 2
    
    # Calculate MAPE (avoiding division by zero)
    merged_df['abs_percent_error'] = np.where(
        merged_df['y'] != 0,
        100.0 * merged_df['abs_error'] / merged_df['y'],
        np.nan
    )
    
    # Calculate metrics
    mape = float(merged_df['abs_percent_error'].dropna().mean()) if not merged_df['abs_percent_error'].dropna().empty else None
    rmse = float(np.sqrt(merged_df['squared_error'].mean())) if not merged_df.empty else None
    mae = float(merged_df['abs_error'].mean()) if not merged_df.empty else None
    
    return {'mape': mape, 'rmse': rmse, 'mae': mae}

def create_no_data_chart():
    """Create a chart for insufficient data scenario"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.5, 'Insufficient data for forecast\n(Need at least 3 data points)', 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=16, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax.set_title('Sales Forecast - Insufficient Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper']), graphic, None

def create_flat_forecast_chart(df, periods, frequency):
    """Create a forecast chart for flat/constant data"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot historical data
    ax.plot(df['ds'], df['y'], 'ko-', label='Historical Data', linewidth=2, markersize=6)
    
    # Create simple flat forecast
    last_date = df['ds'].max()
    avg_value = df['y'].mean()
    
    if frequency == 'W':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), 
                                   periods=periods, freq='W')
    else:
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=periods, freq='MS')
    
    # Plot forecast as horizontal line
    ax.plot(future_dates, [avg_value] * periods, 'b--', 
            label='Forecast (Flat Trend)', linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales Quantity', fontsize=12)
    ax.set_title('Sales Forecast - Constant Trend', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    # Create simple forecast dataframe
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': [avg_value] * periods,
        'yhat_lower': [max(0, avg_value * 0.8)] * periods,
        'yhat_upper': [avg_value * 1.2] * periods
    })
    
    return forecast_df, graphic, None

def create_forecast_chart(model, forecast, df, accuracy_metrics):
    """Create an enhanced forecast visualization"""
    fig = plt.figure(figsize=(14, 10))
    
    # Main forecast plot
    ax1 = plt.subplot(2, 1, 1)
    model.plot(forecast, ax=ax1)
    ax1.set_title('Sales Forecast with Confidence Intervals', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Sales Quantity', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add accuracy metrics if available
    if accuracy_metrics and accuracy_metrics.get('mape') is not None:
        metrics_text = (
            f"Model Accuracy Metrics:\n"
            f"MAPE: {accuracy_metrics['mape']:.1f}%\n"
            f"RMSE: {accuracy_metrics['rmse']:.1f}\n"
            f"MAE: {accuracy_metrics['mae']:.1f}"
        )
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Components plot (if model has components)
    try:
        ax2 = plt.subplot(2, 1, 2)
        components = model.predict(model.history)
        ax2.plot(components['ds'], components['trend'], label='Trend', linewidth=2)
        if 'yearly' in components.columns:
            ax2.plot(components['ds'], components['yearly'], label='Yearly Seasonality', alpha=0.7)
        if 'monthly' in components.columns:
            ax2.plot(components['ds'], components['monthly'], label='Monthly Seasonality', alpha=0.7)
        ax2.set_title('Forecast Components', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Component Value', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    except:
        # If components plot fails, just use a simple summary
        ax2 = plt.subplot(2, 1, 2)
        ax2.text(0.5, 0.5, f'Historical Data Points: {len(df)}\nForecast Points: {len(forecast)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Forecast Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

def create_error_chart(error_message):
    """Create a chart showing error information"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.5, f'Forecasting Error:\n{error_message}', 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax.set_title('Sales Forecast - Error', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper']), graphic, None

# ------------------

# def forecast_with_prophet(df, periods=12, frequency='W'):
#     """
#     Improved Prophet forecasting with better accuracy calculation
#     """
#     # Check if DataFrame is empty or has insufficient data
#     if df.empty or len(df) < 3:
#         return create_no_data_chart()
    
#     # Check for sufficient variability in the data
#     if df['y'].std() == 0:
#         return create_flat_forecast_chart(df, periods, frequency)
    
#     # Configure Prophet model (same as your existing code)
#     model_params = {
#         'changepoint_prior_scale': 0.05,
#         'seasonality_prior_scale': 10.0,
#         'holidays_prior_scale': 10.0,
#         'seasonality_mode': 'additive',
#         'interval_width': 0.80,
#         'mcmc_samples': 0,
#     }
    
#     # Adjust seasonality based on frequency and data length
#     if frequency == 'W':
#         model_params.update({
#             'yearly_seasonality': len(df) >= 52,
#             'weekly_seasonality': False,
#             'daily_seasonality': False,
#         })
#         if len(df) >= 8:
#             model_params['yearly_seasonality'] = False
#     elif frequency == 'M':
#         model_params.update({
#             'yearly_seasonality': len(df) >= 24,
#             'weekly_seasonality': False,
#             'daily_seasonality': False,
#         })
    
#     try:
#         # Initialize and train Prophet model
#         model = Prophet(**model_params)
        
#         # Add custom seasonalities if we have enough data
#         if frequency == 'W' and len(df) >= 8:
#             model.add_seasonality(name='monthly', period=30.5/7, fourier_order=5)
        
#         # Fit model on all historical data
#         model.fit(df)
        
#         # Create future dataframe for prediction
#         future = model.make_future_dataframe(periods=periods, freq=frequency)
        
#         # Generate forecast
#         forecast = model.predict(future)
        
#         # Ensure forecasted values are non-negative
#         forecast['yhat'] = forecast['yhat'].clip(lower=0)
#         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
#         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
#         # Calculate accuracy metrics on historical period
#         accuracy_metrics = calculate_in_sample_accuracy(df, forecast)
        
#         # Create visualization
#         graphic = create_forecast_chart(model, forecast, df, accuracy_metrics)
        
#         return forecast, graphic, accuracy_metrics
        
#     except Exception as e:
#         print(f"Error in Prophet forecasting: {str(e)}")
#         return create_error_chart(str(e))

def calculate_in_sample_accuracy(actual_df, forecast_df):
    """
    Calculate accuracy metrics using in-sample (fitted values) comparison
    """
    if actual_df.empty or forecast_df.empty:
        return {'mape': None, 'rmse': None, 'mae': None}
    
    # Get only the historical period from forecast
    historical_forecast = forecast_df[
        forecast_df['ds'].isin(actual_df['ds'])
    ].copy()
    
    if historical_forecast.empty:
        # Try approximate matching if exact dates don't match
        return calculate_approximate_accuracy(actual_df, forecast_df)
    
    # Merge actual and forecasted data
    merged_df = pd.merge(
        actual_df, 
        historical_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='inner'
    )
    
    if merged_df.empty or len(merged_df) < 2:
        return {'mape': None, 'rmse': None, 'mae': None}
    
    # Calculate errors
    merged_df['error'] = merged_df['y'] - merged_df['yhat']
    merged_df['abs_error'] = abs(merged_df['error'])
    merged_df['squared_error'] = merged_df['error'] ** 2
    
    # Calculate MAPE (avoiding division by zero)
    merged_df['abs_percent_error'] = np.where(
        merged_df['y'] != 0,
        100.0 * merged_df['abs_error'] / merged_df['y'],
        np.nan
    )
    
    # Calculate metrics
    valid_mape = merged_df['abs_percent_error'].dropna()
    mape = float(valid_mape.mean()) if len(valid_mape) > 0 else None
    rmse = float(np.sqrt(merged_df['squared_error'].mean()))
    mae = float(merged_df['abs_error'].mean())
    
    return {'mape': mape, 'rmse': rmse, 'mae': mae}

def calculate_approximate_accuracy(actual_df, forecast_df):
    """
    Calculate accuracy with approximate date matching for misaligned dates
    """
    if len(actual_df) != len(forecast_df[:len(actual_df)]):
        return {'mape': None, 'rmse': None, 'mae': None}
    
    # Take first N forecast points where N = length of actual data
    forecast_subset = forecast_df.head(len(actual_df)).copy()
    
    # Calculate errors using positional matching
    errors = actual_df['y'].values - forecast_subset['yhat'].values
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Calculate MAPE
    valid_actuals = actual_df['y'].values[actual_df['y'].values != 0]
    valid_forecast = forecast_subset['yhat'].values[actual_df['y'].values != 0]
    
    if len(valid_actuals) > 0:
        mape = float(100.0 * np.mean(np.abs(valid_actuals - valid_forecast) / valid_actuals))
    else:
        mape = None
    
    rmse = float(np.sqrt(np.mean(squared_errors)))
    mae = float(np.mean(abs_errors))
    
    return {'mape': mape, 'rmse': rmse, 'mae': mae}

def debug_accuracy_calculation(item_id=None):
    """
    Debug function to check why accuracy metrics are None
    """
    print("=== DEBUG: Accuracy Calculation ===")
    
    # Get data
    weekly_df = prepare_time_series_data(item_id=item_id, period='weekly')
    print(f"Weekly data shape: {weekly_df.shape}")
    print(f"Weekly data:\n{weekly_df.head()}")
    
    if len(weekly_df) >= 3:
        try:
            # Try forecasting
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            model.fit(weekly_df)
            
            future = model.make_future_dataframe(periods=12, freq='W')
            forecast = model.predict(future)
            
            print(f"Forecast shape: {forecast.shape}")
            print(f"Historical forecast dates: {forecast['ds'][:len(weekly_df)].tolist()}")
            print(f"Actual dates: {weekly_df['ds'].tolist()}")
            
            # Check date alignment
            common_dates = set(weekly_df['ds']).intersection(set(forecast['ds']))
            print(f"Common dates: {len(common_dates)}")
            
            # Try accuracy calculation
            accuracy = calculate_in_sample_accuracy(weekly_df, forecast)
            print(f"Accuracy metrics: {accuracy}")
            
        except Exception as e:
            print(f"Error in debug: {str(e)}")
    
    return weekly_df

# --------------
def predict_stockout(item_id, forecast_df):
    """
    Fixed stockout prediction - calculates when stock will run out
    """
    # Check if forecast is empty
    if forecast_df.empty:
        return None
        
    # Get current stock level
    try:
        item = Item.objects.get(id=item_id)
        current_stock = item.quantity
        
        # Extract only future forecast values (not historical fitted values)
        today = timezone.now().date()
        future_forecast = forecast_df[forecast_df['ds'].dt.date > today].copy()
        
        if future_forecast.empty:
            return None
        
        # Sort by date to ensure proper order
        future_forecast = future_forecast.sort_values('ds').reset_index(drop=True)
        
        # Calculate running stock depletion
        running_stock = current_stock
        stockout_date = None
        
        for index, row in future_forecast.iterrows():
            # Subtract forecasted sales from running stock
            predicted_sales = max(0, row['yhat'])  # Ensure non-negative
            running_stock -= predicted_sales
            
            # Check if stock runs out
            if running_stock <= 0:
                stockout_date = row['ds']
                break
        
        return stockout_date
        
    except Item.DoesNotExist:
        return None

def forecast_with_prophet(df, periods=12, frequency='W'):
    """
    Improved Prophet forecasting with better parameter tuning for retail data
    """
    # Check if DataFrame is empty or has insufficient data
    if df.empty or len(df) < 3:
        return create_no_data_chart()
    
    # Check for sufficient variability in the data
    if df['y'].std() == 0:
        return create_flat_forecast_chart(df, periods, frequency)
    
    # Enhanced data preprocessing
    df_processed = df.copy()
    
    # Remove extreme outliers more conservatively for weekly data
    if len(df_processed) > 4:
        Q1 = df_processed['y'].quantile(0.25)
        Q3 = df_processed['y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 1.5 * IQR)  # More conservative outlier removal
        upper_bound = Q3 + 2.0 * IQR  # Allow higher upper bound for sales spikes
        df_processed = df_processed[
            (df_processed['y'] >= lower_bound) & (df_processed['y'] <= upper_bound)
        ]
    
    # If we removed too much data, use original
    if len(df_processed) < 3:
        df_processed = df.copy()
    
    # Configure Prophet with more conservative parameters for weekly data
    if frequency == 'W':
        model_params = {
            'changepoint_prior_scale': 0.01,    # Much more conservative for weekly
            'seasonality_prior_scale': 1.0,     # Reduced seasonality flexibility
            'holidays_prior_scale': 1.0,
            'seasonality_mode': 'additive',
            'interval_width': 0.80,
            'mcmc_samples': 0,
            'yearly_seasonality': False,        # Disable yearly for weekly
            'weekly_seasonality': False,        # Disable weekly seasonality for weekly aggregated data
            'daily_seasonality': False,
        }
        
        # Only add monthly pattern if we have enough data
        add_monthly_seasonality = len(df_processed) >= 12  # At least 3 months
        
    else:  # Monthly
        model_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 5.0,
            'holidays_prior_scale': 5.0,
            'seasonality_mode': 'additive',
            'interval_width': 0.80,
            'mcmc_samples': 0,
            'yearly_seasonality': len(df_processed) >= 24,
            'weekly_seasonality': False,
            'daily_seasonality': False,
        }
        add_monthly_seasonality = False
    
    try:
        # Initialize Prophet model
        model = Prophet(**model_params)
        
        # Add custom seasonalities with more conservative parameters
        if frequency == 'W' and add_monthly_seasonality:
            model.add_seasonality(
                name='monthly', 
                period=30.5/7,  # Monthly pattern in weeks
                fourier_order=3,  # Reduced from 5 to 3 for less overfitting
                prior_scale=1.0   # Conservative prior
            )
        
        # Fit model
        model.fit(df_processed)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=frequency)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Ensure forecasted values are non-negative and reasonable
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        # Apply smoothing to reduce volatility in weekly forecasts
        if frequency == 'W' and len(forecast) > 3:
            # Apply simple moving average smoothing to reduce noise
            window_size = min(3, len(forecast) // 4)
            if window_size > 1:
                forecast['yhat'] = forecast['yhat'].rolling(
                    window=window_size, 
                    center=True, 
                    min_periods=1
                ).mean()
        
        # Calculate accuracy metrics on historical period
        accuracy_metrics = calculate_improved_accuracy(df_processed, forecast)
        
        # Create visualization
        graphic = create_forecast_chart(model, forecast, df_processed, accuracy_metrics)
        
        return forecast, graphic, accuracy_metrics
        
    except Exception as e:
        print(f"Error in Prophet forecasting: {str(e)}")
        return create_error_chart(str(e))

def calculate_improved_accuracy(actual_df, forecast_df):
    """
    Improved accuracy calculation with better handling of weekly volatility
    """
    if actual_df.empty or forecast_df.empty:
        return {'mape': None, 'rmse': None, 'mae': None}
    
    # Get historical forecast (fitted values)
    historical_forecast = forecast_df[
        forecast_df['ds'].isin(actual_df['ds'])
    ].copy()
    
    if historical_forecast.empty:
        return calculate_approximate_accuracy(actual_df, forecast_df)
    
    # Merge actual and forecasted data
    merged_df = pd.merge(
        actual_df, 
        historical_forecast[['ds', 'yhat']], 
        on='ds', 
        how='inner'
    )
    
    if merged_df.empty or len(merged_df) < 2:
        return {'mape': None, 'rmse': None, 'mae': None}
    
    # Calculate errors
    merged_df['error'] = merged_df['y'] - merged_df['yhat']
    merged_df['abs_error'] = abs(merged_df['error'])
    merged_df['squared_error'] = merged_df['error'] ** 2
    
    # Improved MAPE calculation that handles zero values better
    # For retail data, use a small epsilon to avoid division by zero
    epsilon = 0.1  # Small value to avoid division by zero
    merged_df['adj_actual'] = merged_df['y'] + epsilon
    merged_df['percent_error'] = 100.0 * merged_df['abs_error'] / merged_df['adj_actual']
    
    # Alternative: Symmetric MAPE (SMAPE) which is more robust
    merged_df['smape'] = 100.0 * merged_df['abs_error'] / (
        (abs(merged_df['y']) + abs(merged_df['yhat'])) / 2 + epsilon
    )
    
    # Calculate metrics
    mape = float(merged_df['percent_error'].mean())
    smape = float(merged_df['smape'].mean())
    rmse = float(np.sqrt(merged_df['squared_error'].mean()))
    mae = float(merged_df['abs_error'].mean())
    
    # Return SMAPE instead of MAPE for better interpretation
    return {
        'mape': smape,  # Using SMAPE as it's more robust
        'rmse': rmse, 
        'mae': mae,
        'traditional_mape': mape  # Keep original for comparison
    }

def prepare_time_series_data(item_id=None, period='weekly'):
    """
    Improved time series data preparation with better handling of weekly volatility
    """
    # Query transactions, filter by item if specified
    queryset = Transaction.objects.all()
    if item_id:
        queryset = queryset.filter(item_id=item_id)
    
    # Group by time period and sum quantities
    if period == 'weekly':
        data = queryset.annotate(
            week=models.functions.TruncWeek('timestamp')
        ).values('week').annotate(
            total_quantity=Sum('quantity')
        ).order_by('week')
        
        df = pd.DataFrame(list(data))
        if df.empty:
            return pd.DataFrame(columns=['ds', 'y'])
            
        df.rename(columns={'week': 'ds', 'total_quantity': 'y'}, inplace=True)
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        # Fill gaps in weekly data
        df = fill_time_gaps(df, freq='W')
        
        # Apply additional smoothing for weekly data to reduce noise
        if len(df) > 2:
            # Apply light smoothing to reduce weekly volatility
            df['y_smoothed'] = df['y'].rolling(window=2, center=True, min_periods=1).mean()
            # Use original for zero weeks, smoothed for others
            df['y'] = df['y_smoothed']
            df = df.drop('y_smoothed', axis=1)
        
    elif period == 'monthly':
        data = queryset.annotate(
            month=models.functions.TruncMonth('timestamp')
        ).values('month').annotate(
            total_quantity=Sum('quantity')
        ).order_by('month')
        
        df = pd.DataFrame(list(data))
        if df.empty:
            return pd.DataFrame(columns=['ds', 'y'])
            
        df.rename(columns={'month': 'ds', 'total_quantity': 'y'}, inplace=True)
        df['ds'] = df['ds'].dt.tz_localize(None)
        
        # Fill gaps in monthly data
        df = fill_time_gaps(df, freq='M')
    
    # Clean the data with more conservative approach
    df = clean_time_series_data_improved(df, period)
    
    return df

def clean_time_series_data_improved(df, period='weekly'):
    """
    Improved data cleaning with period-specific handling
    """
    if df.empty or len(df) < 2:
        return df
    
    # Remove negative values
    df = df[df['y'] >= 0].copy()
    
    if period == 'weekly':
        # More conservative outlier removal for weekly data
        if len(df) > 4:
            # Use quantile-based approach instead of standard deviation
            Q1 = df['y'].quantile(0.10)  # More conservative
            Q3 = df['y'].quantile(0.90)
            IQR = Q3 - Q1
            
            # Allow for reasonable sales spikes
            upper_threshold = Q3 + 2.0 * IQR  # More lenient upper bound
            lower_threshold = max(0, Q1 - 1.0 * IQR)
            
            df = df[(df['y'] >= lower_threshold) & (df['y'] <= upper_threshold)].copy()
    
    else:  # Monthly data
        # Standard outlier removal for monthly data
        if len(df) > 3:
            mean_val = df['y'].mean()
            std_val = df['y'].std()
            if std_val > 0:
                df = df[abs(df['y'] - mean_val) <= 2.5 * std_val].copy()
    
    # Ensure we still have enough data points
    if len(df) < 2:
        return pd.DataFrame(columns=['ds', 'y'])
    
    return df

def fill_time_gaps(df, freq='W'):
    """
    Improved gap filling with better handling of missing periods
    """
    if df.empty:
        return df
    
    # Create a complete date range
    start_date = df['ds'].min()
    end_date = df['ds'].max()
    
    if freq == 'W':
        date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    elif freq == 'M':
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    else:
        return df
    
    # Create complete dataframe with all dates
    complete_df = pd.DataFrame({'ds': date_range})
    
    # Merge with existing data
    df = complete_df.merge(df, on='ds', how='left')
    
    # Improved missing value handling
    if freq == 'W':
        # For weekly data, use forward fill for short gaps, zero for longer gaps
        df['y'] = df['y'].fillna(method='ffill', limit=1)  # Forward fill max 1 week
        df['y'] = df['y'].fillna(0)  # Fill remaining with 0
    else:
        # For monthly data, use interpolation for single missing months
        df['y'] = df['y'].interpolate(method='linear', limit=1)
        df['y'] = df['y'].fillna(0)
    
    return df



# Advanced forecast metrics
class AdvancedForecastingMetrics:
    """
    Advanced forecasting evaluation metrics for sales prediction
    """
    
    @staticmethod
    def calculate_smape(y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-8  # Small value to avoid division by zero
        
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        
        smape = np.mean(numerator / denominator) * 100
        return smape
    
    @staticmethod
    def calculate_mase(y_true, y_pred, y_train):
        """Mean Absolute Scaled Error"""
        y_true, y_pred, y_train = np.array(y_true), np.array(y_pred), np.array(y_train)
        
        # Calculate MAE of forecast
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        
        # Calculate MAE of naive forecast (seasonal naive for sales data)
        if len(y_train) > 1:
            # Use simple naive forecast (last period = current period)
            naive_errors = np.abs(np.diff(y_train))
            mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        else:
            mae_naive = 1.0
        
        # Avoid division by zero
        mae_naive = max(mae_naive, 1e-8)
        
        mase = mae_forecast / mae_naive
        return mase
    
    @staticmethod
    def calculate_crps_empirical(y_true, y_pred_lower, y_pred_upper, y_pred_mean):
        """
        Simplified CRPS calculation using prediction intervals
        """
        y_true = np.array(y_true)
        y_pred_lower = np.array(y_pred_lower)
        y_pred_upper = np.array(y_pred_upper)
        y_pred_mean = np.array(y_pred_mean)
        
        crps_scores = []
        
        for i in range(len(y_true)):
            # Simplified CRPS approximation using prediction intervals
            # This assumes a normal distribution between lower and upper bounds
            true_val = y_true[i]
            pred_mean = y_pred_mean[i]
            pred_std = (y_pred_upper[i] - y_pred_lower[i]) / (2 * 1.96)  # Approximate std from 95% CI
            
            if pred_std > 0:
                # CRPS for normal distribution
                z = (true_val - pred_mean) / pred_std
                crps = pred_std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
            else:
                crps = abs(true_val - pred_mean)
            
            crps_scores.append(crps)
        
        return np.mean(crps_scores)
    
    @staticmethod
    def calculate_prediction_intervals_coverage(y_true, y_pred_lower, y_pred_upper, confidence_level=0.95):
        """
        Calculate prediction interval coverage
        """
        y_true = np.array(y_true)
        y_pred_lower = np.array(y_pred_lower)
        y_pred_upper = np.array(y_pred_upper)
        
        # Check if actual values fall within prediction intervals
        within_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        coverage = np.mean(within_interval)
        
        # Calculate average interval width
        avg_width = np.mean(y_pred_upper - y_pred_lower)
        
        # Calculate normalized interval width
        mean_actual = np.mean(y_true) if np.mean(y_true) > 0 else 1
        normalized_width = avg_width / mean_actual
        
        return {
            'coverage': coverage,
            'expected_coverage': confidence_level,
            'avg_width': avg_width,
            'normalized_width': normalized_width,
            'coverage_difference': abs(coverage - confidence_level)
        }

def calculate_item_performance_metrics_enhanced(item_id=None, period='monthly', months=6):
    """
    Enhanced version of your existing function with advanced metrics
    """
    end_date = timezone.now().date()
    start_date = end_date - datetime.timedelta(days=30*months)
    
    # Query base (same as your existing code)
    queryset = Transaction.objects.filter(timestamp__gte=start_date)
    if item_id:
        queryset = queryset.filter(item_id=item_id)
        try:
            item = Item.objects.get(id=item_id)
            current_stock = item.quantity
        except Item.DoesNotExist:
            current_stock = 0
    else:
        current_stock = Item.objects.aggregate(Sum('quantity'))['quantity__sum'] or 0
    
    # Basic metrics (same as your existing code)
    total_sales = queryset.aggregate(Sum('quantity'))['quantity__sum'] or 0
    days_in_period = (end_date - start_date).days
    if days_in_period == 0:
        days_in_period = 1
    
    sales_velocity_daily = total_sales / days_in_period
    sales_velocity_weekly = sales_velocity_daily * 7
    sales_velocity_monthly = sales_velocity_daily * 30
    
    days_of_supply = None
    if sales_velocity_daily > 0:
        days_of_supply = current_stock / sales_velocity_daily
    
    inventory_turnover = None
    if current_stock > 0:
        inventory_turnover = (total_sales / days_in_period * 365) / current_stock
    
    # NEW: Add category classification
    # Determine if this is high-selling or low-selling item
    if item_id:
        # Get overall sales percentile
        all_items_sales = []
        for item in Item.objects.all():
            item_sales = Transaction.objects.filter(
                item_id=item.id, 
                timestamp__gte=start_date
            ).aggregate(Sum('quantity'))['quantity__sum'] or 0
            all_items_sales.append(item_sales)
        
        if all_items_sales:
            sales_percentile = stats.percentileofscore(all_items_sales, total_sales)
            category = 'high_selling' if sales_percentile >= 50 else 'low_selling'
        else:
            category = 'unknown'
    else:
        category = 'overall'
    
    return {
        'sales_velocity_daily': sales_velocity_daily,
        'sales_velocity_weekly': sales_velocity_weekly,
        'sales_velocity_monthly': sales_velocity_monthly,
        'days_of_supply': days_of_supply,
        'inventory_turnover': inventory_turnover,
        'total_sales_period': total_sales,
        'current_stock': current_stock,
        'category': category,  # NEW
        'sales_percentile': sales_percentile if item_id else None  # NEW
    }

def forecast_with_prophet_enhanced(df, periods=12, frequency='W', item_id=None):
    """
    Enhanced Prophet forecasting with advanced metrics
    """
    # Check if DataFrame is empty or has insufficient data
    if df.empty or len(df) < 3:
        return create_no_data_chart()
    
    # Check for sufficient variability in the data
    if df['y'].std() == 0:
        return create_flat_forecast_chart(df, periods, frequency)
    
    # Split data for validation (80% train, 20% test)
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point].copy()
    test_df = df.iloc[split_point:].copy()
    
    # If test set is too small, use all data for training
    if len(test_df) < 2:
        train_df = df.copy()
        test_df = pd.DataFrame()
    
    # Configure Prophet (same as your existing code)
    model_params = {
        'changepoint_prior_scale': 0.01,
        'seasonality_prior_scale': 1.0,
        'holidays_prior_scale': 1.0,
        'seasonality_mode': 'additive',
        'interval_width': 0.80,
        'mcmc_samples': 0,
        'yearly_seasonality': False,
        'weekly_seasonality': False,
        'daily_seasonality': False,
    }
    
    try:
        # Initialize Prophet model
        model = Prophet(**model_params)
        
        # Add custom seasonalities
        if frequency == 'W' and len(train_df) >= 12:
            model.add_seasonality(
                name='monthly', 
                period=30.5/7,
                fourier_order=3,
                prior_scale=1.0
            )
        
        # Fit model on training data
        model.fit(train_df)
        
        # Create future dataframe for full forecast
        future_full = model.make_future_dataframe(periods=periods, freq=frequency)
        forecast_full = model.predict(future_full)
        
        # Ensure non-negative forecasts
        forecast_full['yhat'] = forecast_full['yhat'].clip(lower=0)
        forecast_full['yhat_lower'] = forecast_full['yhat_lower'].clip(lower=0)
        forecast_full['yhat_upper'] = forecast_full['yhat_upper'].clip(lower=0)
        
        # NEW: Calculate advanced metrics if we have test data
        advanced_metrics = {}
        if not test_df.empty:
            # Predict on test set
            test_future = pd.DataFrame({'ds': test_df['ds']})
            test_forecast = model.predict(test_future)
            
            # Ensure arrays have same length
            min_len = min(len(test_df), len(test_forecast))
            y_true = test_df['y'].iloc[:min_len].values
            y_pred = test_forecast['yhat'].iloc[:min_len].values
            y_pred_lower = test_forecast['yhat_lower'].iloc[:min_len].values
            y_pred_upper = test_forecast['yhat_upper'].iloc[:min_len].values
            
            if len(y_true) > 0:
                # Initialize advanced metrics calculator
                metrics_calc = AdvancedForecastingMetrics()
                
                # Calculate advanced metrics
                advanced_metrics = {
                    'smape': metrics_calc.calculate_smape(y_true, y_pred),
                    'mase': metrics_calc.calculate_mase(y_true, y_pred, train_df['y'].values),
                    'crps': metrics_calc.calculate_crps_empirical(y_true, y_pred_lower, y_pred_upper, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'traditional_mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                }
                
                # Calculate prediction interval coverage for different confidence levels
                confidence_levels = [0.8, 0.9, 0.95]  # Assuming 80% interval from Prophet
                advanced_metrics['prediction_intervals'] = {}
                
                for conf_level in confidence_levels:
                    if conf_level == 0.8:  # Use Prophet's default 80% interval
                        coverage_metrics = metrics_calc.calculate_prediction_intervals_coverage(
                            y_true, y_pred_lower, y_pred_upper, conf_level
                        )
                        advanced_metrics['prediction_intervals'][f'{int(conf_level*100)}%'] = coverage_metrics
        
        # Calculate overall accuracy on full dataset (in-sample)
        in_sample_forecast = forecast_full[forecast_full['ds'].isin(df['ds'])]
        if len(in_sample_forecast) == len(df):
            metrics_calc = AdvancedForecastingMetrics()
            
            y_true_full = df['y'].values
            y_pred_full = in_sample_forecast['yhat'].values
            
            # Add in-sample metrics
            advanced_metrics['in_sample'] = {
                'smape': metrics_calc.calculate_smape(y_true_full, y_pred_full),
                'rmse': np.sqrt(mean_squared_error(y_true_full, y_pred_full)),
                'mae': mean_absolute_error(y_true_full, y_pred_full)
            }
        
        # Create visualization (enhanced version of your existing function)
        graphic = create_enhanced_forecast_chart(model, forecast_full, df, advanced_metrics)
        
        return forecast_full, graphic, advanced_metrics
        
    except Exception as e:
        print(f"Error in enhanced Prophet forecasting: {str(e)}")
        return create_error_chart(str(e))

def create_enhanced_forecast_chart(model, forecast, df, advanced_metrics):
    """
    Enhanced version of your forecast chart with metrics display
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Main forecast plot
    ax1 = plt.subplot(3, 1, 1)
    model.plot(forecast, ax=ax1)
    ax1.set_title('Sales Forecast with Confidence Intervals', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Sales Quantity', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add advanced metrics text box
    if advanced_metrics:
        metrics_text = "Advanced Metrics:\n"
        if 'smape' in advanced_metrics:
            metrics_text += f"sMAPE: {advanced_metrics['smape']:.2f}%\n"
        if 'mase' in advanced_metrics:
            metrics_text += f"MASE: {advanced_metrics['mase']:.3f}\n"
        if 'crps' in advanced_metrics:
            metrics_text += f"CRPS: {advanced_metrics['crps']:.3f}\n"
        if 'rmse' in advanced_metrics:
            metrics_text += f"RMSE: {advanced_metrics['rmse']:.2f}\n"
        if 'mae' in advanced_metrics:
            metrics_text += f"MAE: {advanced_metrics['mae']:.2f}\n"
        
        # Add prediction interval coverage
        if 'prediction_intervals' in advanced_metrics:
            metrics_text += "\nPrediction Intervals:\n"
            for level, metrics in advanced_metrics['prediction_intervals'].items():
                coverage = metrics['coverage'] * 100
                expected = metrics['expected_coverage'] * 100
                metrics_text += f"{level}: {coverage:.1f}% (target: {expected:.0f}%)\n"
        
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Components plot
    try:
        ax2 = plt.subplot(3, 1, 2)
        components = model.predict(model.history)
        ax2.plot(components['ds'], components['trend'], label='Trend', linewidth=2)
        if 'monthly' in components.columns:
            ax2.plot(components['ds'], components['monthly'], label='Monthly Seasonality', alpha=0.7)
        ax2.set_title('Forecast Components', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Component Value', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    except:
        ax2 = plt.subplot(3, 1, 2)
        ax2.text(0.5, 0.5, f'Historical Data Points: {len(df)}\nForecast Points: {len(forecast)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Forecast Summary', fontsize=14, fontweight='bold')
    
    # Residuals plot (if we have test metrics)
    ax3 = plt.subplot(3, 1, 3)
    if advanced_metrics and 'in_sample' in advanced_metrics:
        # Plot residuals for in-sample forecast
        in_sample_forecast = forecast[forecast['ds'].isin(df['ds'])]
        if len(in_sample_forecast) == len(df):
            residuals = df['y'].values - in_sample_forecast['yhat'].values
            ax3.plot(df['ds'], residuals, 'o-', alpha=0.7, label='Residuals')
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax3.set_title('Forecast Residuals', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=12)
            ax3.set_ylabel('Residual', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Residuals plot not available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'Advanced metrics calculated on validation set', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=12)
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')

# Enhanced analytics dashboard
def analytics_dashboard_enhanced(request):
    """
    Enhanced analytics dashboard with category-based analysis
    """
    # Get all items
    items = Item.objects.all()
    
    # Initialize category results
    category_results = {
        'high_selling': {'items': [], 'metrics': [], 'total_sales': 0},
        'low_selling': {'items': [], 'metrics': [], 'total_sales': 0}
    }
    
    # Process each item with enhanced metrics
    all_items_metrics = []
    
    for item in items:
        try:
            # Get item performance metrics with category
            item_metrics = calculate_item_performance_metrics_enhanced(item_id=item.id)
            
            # Get forecasting data
            item_df = prepare_time_series_data(item_id=item.id, period='weekly')
            
            if len(item_df) >= 3:
                # Run enhanced forecasting
                forecast, chart, advanced_metrics = forecast_with_prophet_enhanced(
                    item_df, periods=12, frequency='W', item_id=item.id
                )
                
                # Combine basic and advanced metrics
                combined_metrics = {
                    **item_metrics,
                    'advanced_metrics': advanced_metrics,
                    'forecast': forecast,
                    'chart': chart
                }
                
                # Add to category
                category = item_metrics['category']
                if category in category_results:
                    category_results[category]['items'].append({
                        'item': item,
                        'metrics': combined_metrics
                    })
                    category_results[category]['total_sales'] += item_metrics['total_sales_period']
                    
                    # Collect advanced metrics for aggregation
                    if advanced_metrics:
                        category_results[category]['metrics'].append(advanced_metrics)
                
                all_items_metrics.append({
                    'item': item,
                    'metrics': combined_metrics
                })
                
        except Exception as e:
            print(f"Error processing item {item.id} ({item.name}): {str(e)}")
    
    # Calculate aggregated metrics for each category
    for category in category_results:
        metrics_list = category_results[category]['metrics']
        if metrics_list:
            # Calculate average metrics
            avg_metrics = {}
            for metric_name in ['smape', 'mase', 'crps', 'rmse', 'mae']:
                values = [m[metric_name] for m in metrics_list if metric_name in m and m[metric_name] is not None]
                if values:
                    avg_metrics[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            
            category_results[category]['aggregated_metrics'] = avg_metrics
    
    # Calculate overall performance
    try:
        overall_performance = calculate_item_performance_metrics_enhanced(period='monthly')
    except Exception as e:
        print(f"Error calculating overall performance: {str(e)}")
        overall_performance = {}
    
    # Get latest transactions
    latest_transactions = Transaction.objects.all().order_by('-timestamp')[:10]
    
    # Prepare context with enhanced data
    context = {
        'category_results': category_results,
        'overall_performance': overall_performance,
        'all_items_metrics': all_items_metrics[:20],  # Limit display
        'latest_transactions': latest_transactions,
        'items': items,
        'total_items': items.count(),
        'low_stock_items': items.filter(quantity__lt=10).count(),
        'advanced_metrics_enabled': True,
        # Summary statistics
        'high_selling_count': len(category_results['high_selling']['items']),
        'low_selling_count': len(category_results['low_selling']['items']),
        'total_sales_high': category_results['high_selling']['total_sales'],
        'total_sales_low': category_results['low_selling']['total_sales'],
    }
    
    return render(request, 'AApp/analytics/enanalytics.html', context)

# Enhanced item analytics view
def item_analytics_view_enhanced(request, item_id):
    """
    Enhanced item analytics view with advanced metrics
    """
    try:
        # Get the item
        item = Item.objects.get(id=item_id)
        
        # Prepare time series data
        weekly_df = prepare_time_series_data(item_id=item_id, period='weekly')
        monthly_df = prepare_time_series_data(item_id=item_id, period='monthly')
        
        # Generate enhanced forecasts
        weekly_results = None
        monthly_results = None
        
        if len(weekly_df) >= 3:
            try:
                weekly_forecast, weekly_chart, weekly_metrics = forecast_with_prophet_enhanced(
                    weekly_df, periods=12, frequency='W', item_id=item_id
                )
                weekly_results = {
                    'forecast': weekly_forecast,
                    'chart': weekly_chart,
                    'metrics': weekly_metrics
                }
            except Exception as e:
                print(f"Error in weekly forecast for item {item.name}: {str(e)}")
        
        if len(monthly_df) >= 3:
            try:
                monthly_forecast, monthly_chart, monthly_metrics = forecast_with_prophet_enhanced(
                    monthly_df, periods=6, frequency='M', item_id=item_id
                )
                monthly_results = {
                    'forecast': monthly_forecast,
                    'chart': monthly_chart,
                    'metrics': monthly_metrics
                }
            except Exception as e:
                print(f"Error in monthly forecast for item {item.name}: {str(e)}")
        
        # Calculate enhanced performance metrics
        try:
            performance_metrics = calculate_item_performance_metrics_enhanced(item_id=item_id)
        except Exception as e:
            print(f"Error calculating performance metrics for item {item.name}: {str(e)}")
            performance_metrics = {}
        
        # Predict stockout date
        stockout_date = None
        if weekly_results and weekly_results['forecast'] is not None:
            try:
                stockout_date = predict_stockout(item_id, weekly_results['forecast'])
            except Exception as e:
                print(f"Error predicting stockout for item {item.name}: {str(e)}")
        
        context = {
            'item': item,
            'weekly_results': weekly_results,
            'monthly_results': monthly_results,
            'stockout_date': stockout_date,
            'performance_metrics': performance_metrics,
            'weekly_data_points': len(weekly_df),
            'monthly_data_points': len(monthly_df),
            'advanced_metrics_available': True,
        }
        
        return render(request, 'AApp/analytics/enhanced_item_analytics.html', context)
        
    except Item.DoesNotExist:
        messages.error(request, f'Item with ID {item_id} does not exist.')
        return redirect('items-list')
    except Exception as e:
        messages.error(request, f'Error loading enhanced analytics: {str(e)}')
        return redirect('items-list')

def get_category_comparison_report(request):
    """
    API endpoint to get category comparison report
    """
    if request.method == 'GET':
        try:
            # Get all items and categorize them
            items = Item.objects.all()
            
            report_data = {
                'high_selling': {'count': 0, 'avg_smape': 0, 'avg_mase': 0, 'avg_crps': 0},
                'low_selling': {'count': 0, 'avg_smape': 0, 'avg_mase': 0, 'avg_crps': 0},
                'comparison': {}
            }
            
            high_selling_metrics = []
            low_selling_metrics = []
            
            for item in items:
                try:
                    item_df = prepare_time_series_data(item_id=item.id, period='weekly')
                    if len(item_df) >= 3:
                        _, _, advanced_metrics = forecast_with_prophet_enhanced(
                            item_df, periods=12, frequency='W', item_id=item.id
                        )
                        
                        if advanced_metrics:
                            # Get item category
                            perf_metrics = calculate_item_performance_metrics_enhanced(item_id=item.id)
                            category = perf_metrics.get('category', 'unknown')
                            
                            if category == 'high_selling':
                                high_selling_metrics.append(advanced_metrics)
                            elif category == 'low_selling':
                                low_selling_metrics.append(advanced_metrics)
                                
                except Exception as e:
                    continue
            
            # Calculate averages
            if high_selling_metrics:
                report_data['high_selling'] = {
                    'count': len(high_selling_metrics),
                    'avg_smape': np.mean([m['smape'] for m in high_selling_metrics if 'smape' in m]),
                    'avg_mase': np.mean([m['mase'] for m in high_selling_metrics if 'mase' in m]),
                    'avg_crps': np.mean([m['crps'] for m in high_selling_metrics if 'crps' in m]),
                }
            
            if low_selling_metrics:
                report_data['low_selling'] = {
                    'count': len(low_selling_metrics),
                    'avg_smape': np.mean([m['smape'] for m in low_selling_metrics if 'smape' in m]),
                    'avg_mase': np.mean([m['mase'] for m in low_selling_metrics if 'mase' in m]),
                    'avg_crps': np.mean([m['crps'] for m in low_selling_metrics if 'crps' in m]),
                }
            
            return JsonResponse(report_data)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@require_http_methods(["GET"])
def get_item_advanced_metrics(request, item_id):
    """
    API endpoint to get advanced metrics for a specific item
    """
    try:
        # Get item data
        item_df = prepare_time_series_data(item_id=item_id, period='weekly')
        
        if len(item_df) >= 3:
            # Run enhanced forecasting
            forecast, chart, advanced_metrics = forecast_with_prophet_enhanced(
                item_df, periods=12, frequency='W', item_id=item_id
            )
            
            # Prepare JSON response
            response_data = {
                'item_id': item_id,
                'data_points': len(item_df),
                'metrics': advanced_metrics if advanced_metrics else {},
                'has_forecast': forecast is not None,
                'timestamp': timezone.now().isoformat()
            }
            
            return JsonResponse(response_data)
        else:
            return JsonResponse({
                'error': 'Insufficient data',
                'item_id': item_id,
                'data_points': len(item_df),
                'minimum_required': 3
            }, status=400)
            
    except Item.DoesNotExist:
        return JsonResponse({'error': 'Item not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)








# rev2
# @csrf_exempt
# def update_cart(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             cart_id = data.get('cart_id', 'default')
#             items = data.get('items', {})  
#             total_cost = data.get('total_cost', 0.0)
#             total_weight = data.get('total_weight', 0.0)

#             matched_items = []
#             detected_items = list(items.keys())  
            
#             for item_name, item_details in items.items():
#                 quantity = item_details.get("quantity", 1)  
                
#                 db_items = Item.objects.filter(name__iexact=item_name)
                
#                 if not db_items.exists():
#                     modified_name = item_name[1:]
#                     db_items = Item.objects.filter(name__icontains=modified_name)
                
#                 if db_items.exists():
#                     for db_item in db_items:
#                         matched_items.append({
#                             "name": db_item.name,
#                             "quantity": quantity,  
#                             "price": float(db_item.price),  
#                             "description": db_item.description
#                         })
#                 else:
#                     print(f"Item '{item_name}' (or '{modified_name}') not found in database")

#             # Print matched items
#             if matched_items:
#                 print("\nMatched Items in Database:")
#                 for item in matched_items:
#                     print(f"Name: {item['name']}, Quantity: {item['quantity']}, Price: ₹{item['price']}, Description: {item['description']}")
#             else:
#                 print("\nNo matching items found in database.")
                
#             # Store detected items in cache
#             cache.set('detected_items', detected_items, timeout=None)  
#             cache.set('detected_matched_items', matched_items, timeout=None)
            
#             print(f"Stored in cache - detected_items: {detected_items}")
#             print(f"Stored in cache - matched items count: {len(matched_items)}")

#             return JsonResponse({
#                 'status': 'success',
#                 'message': 'Cart updated successfully',
#                 'matched_items': matched_items  
#             })
#         except Exception as e:
#             print(f"Error in update_cart: {str(e)}")
#             return JsonResponse({
#                 'status': 'error',
#                 'message': str(e)
#             }, status=400)
    
#     return JsonResponse({
#         'status': 'error',
#         'message': 'Only POST requests are allowed'
#     }, status=405)


# @csrf_exempt
# def view_floor(request, floor_name):
#     try:
#         items = Item.objects.all()
#         floor = Floor.objects.get(name=floor_name)
#         shelves = Shelf.objects.filter(floor=floor).values(
#             'cell_index', 'mode', 'item_ids', 'item_names')
        
#         context = {
#             'items': items,
#             'floor': floor, 
#             'shelves': list(shelves)
#             }
#         return render(request, 'AApp/build/viewFloor.html', context)
#     except Floor.DoesNotExist:
#         return render(request, 'AApp/build/viewFloor.html', {'floor_name': floor_name})




