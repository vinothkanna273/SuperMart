from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse
from django.core.cache import cache
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Sum
from django.db import models

import json
import csv
import docx
import urllib.parse
from decimal import Decimal
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
from django.db.models import Sum
from django.db import models
from django.utils import timezone
from decimal import Decimal
import datetime


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


@login_required  
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

@login_required  
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


# -----------------------

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




