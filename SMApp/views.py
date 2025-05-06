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
import pandas as pd
import csv
import docx
import urllib.parse
from decimal import Decimal
from collections import deque
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import io
import base64
# from datetime import datetime, timedelta
import datetime
from django.utils import timezone


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




# analytics
# def prepare_time_series_data(item_id=None, period='weekly'):
#     # Query transactions, filter by item if specified
#     queryset = Transaction.objects.all()
#     if item_id:
#         queryset = queryset.filter(item_id=item_id)
    
#     # Group by time period and sum quantities
#     if period == 'weekly':
#         # Extract week from timestamp and group
#         data = queryset.annotate(
#             week=models.functions.TruncWeek('timestamp')
#         ).values('week').annotate(
#             total_quantity=Sum('quantity')
#         ).order_by('week')
        
#         # Convert to DataFrame
#         df = pd.DataFrame(list(data))
#         df.rename(columns={'week': 'ds', 'total_quantity': 'y'}, inplace=True)
    
#     elif period == 'monthly':
#         # Extract month from timestamp and group
#         data = queryset.annotate(
#             month=models.functions.TruncMonth('timestamp')
#         ).values('month').annotate(
#             total_quantity=Sum('quantity')
#         ).order_by('month')
        
#         # Convert to DataFrame
#         df = pd.DataFrame(list(data))
#         df.rename(columns={'month': 'ds', 'total_quantity': 'y'}, inplace=True)
    
#     return df


# def prepare_time_series_data(item_id=None, period='weekly'):
#     # Query transactions, filter by item if specified
#     queryset = Transaction.objects.all()
#     if item_id:
#         queryset = queryset.filter(item_id=item_id)
    
#     # Group by time period and sum quantities
#     if period == 'weekly':
#         # Extract week from timestamp and group
#         data = queryset.annotate(
#             week=models.functions.TruncWeek('timestamp')
#         ).values('week').annotate(
#             total_quantity=Sum('quantity')
#         ).order_by('week')
        
#         # Convert to DataFrame
#         df = pd.DataFrame(list(data))
#         if df.empty:
#             # Return empty DataFrame with correct columns
#             return pd.DataFrame(columns=['ds', 'y'])
            
#         df.rename(columns={'week': 'ds', 'total_quantity': 'y'}, inplace=True)
        
#         # Remove timezone information
#         df['ds'] = df['ds'].dt.tz_localize(None)
    
#     elif period == 'monthly':
#         # Extract month from timestamp and group
#         data = queryset.annotate(
#             month=models.functions.TruncMonth('timestamp')
#         ).values('month').annotate(
#             total_quantity=Sum('quantity')
#         ).order_by('month')
        
#         # Convert to DataFrame
#         df = pd.DataFrame(list(data))
#         if df.empty:
#             # Return empty DataFrame with correct columns
#             return pd.DataFrame(columns=['ds', 'y'])
            
#         df.rename(columns={'month': 'ds', 'total_quantity': 'y'}, inplace=True)
        
#         # Remove timezone information
#         df['ds'] = df['ds'].dt.tz_localize(None)
    
#     return df

# def forecast_with_prophet(df, periods=12, frequency='W'):
#     # Check if DataFrame is empty or has insufficient data
#     if df.empty or len(df) < 2:  # Prophet needs at least 2 data points
#         # Create a blank image for the chart
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.text(0.5, 0.5, 'Insufficient data for forecast', 
#                 horizontalalignment='center', verticalalignment='center',
#                 transform=ax.transAxes, fontsize=14)
#         plt.close(fig)
        
#         # Convert plot to base64 for embedding in HTML
#         buffer = io.BytesIO()
#         fig.savefig(buffer, format='png')
#         buffer.seek(0)
#         image_png = buffer.getvalue()
#         buffer.close()
        
#         graphic = base64.b64encode(image_png).decode('utf-8')
        
#         # Return empty forecast and the "no data" image
#         return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper']), graphic
    
#     # Initialize and train Prophet model
#     model = Prophet(
#         yearly_seasonality=True,
#         weekly_seasonality=True,
#         daily_seasonality=False,
#         changepoint_prior_scale=0.05
#     )
#     model.fit(df)
    
#     # Create future dataframe for prediction
#     future = model.make_future_dataframe(periods=periods, freq=frequency)
    
#     # Generate forecast
#     forecast = model.predict(future)
    
#     # Create visualization
#     fig = model.plot(forecast)
#     plt.title('Sales Forecast')
    
#     # Convert plot to base64 for embedding in HTML
#     buffer = io.BytesIO()
#     fig.savefig(buffer, format='png')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     buffer.close()
    
#     graphic = base64.b64encode(image_png).decode('utf-8')
    
#     return forecast, graphic


# def predict_stockout(item_id, forecast_df):
#     # Check if forecast is empty
#     if forecast_df.empty:
#         return None
        
#     # Get current stock level
#     try:
#         item = Item.objects.get(id=item_id)
#         current_stock = item.quantity
        
#         # Extract the forecasted values
#         forecast_values = forecast_df[['ds', 'yhat']].copy()
        
#         # Calculate cumulative sales
#         forecast_values['cumulative_sales'] = forecast_values['yhat'].cumsum()
        
#         # Find when cumulative sales exceeds current stock
#         stockout_date = None
#         for index, row in forecast_values.iterrows():
#             if row['cumulative_sales'] >= current_stock:
#                 stockout_date = row['ds']
#                 break
        
#         return stockout_date
#     except Item.DoesNotExist:
#         return None


# def item_analytics_view(request, item_id):
#     # Get the item
#     item = Item.objects.get(id=item_id)
    
#     # Prepare weekly data
#     weekly_df = prepare_time_series_data(item_id=item_id, period='weekly')
#     monthly_df = prepare_time_series_data(item_id=item_id, period='monthly')
    
#     # Generate forecasts
#     weekly_forecast, weekly_chart = forecast_with_prophet(weekly_df, periods=12, frequency='W')
#     monthly_forecast, monthly_chart = forecast_with_prophet(monthly_df, periods=6, frequency='M')
    
#     # Predict stockout date
#     stockout_date = predict_stockout(item_id, weekly_forecast)
    
#     context = {
#         'item': item,
#         'weekly_chart': weekly_chart,
#         'monthly_chart': monthly_chart,
#         'stockout_date': stockout_date,
#     }
    
#     return render(request, 'AApp/analytics/item_analytics.html', context)


# def analytics_dashboard(request):
#     # Get all items
#     items = Item.objects.all()
    
#     # Get overall sales trend
#     overall_df = prepare_time_series_data(period='weekly')
#     overall_forecast, overall_chart = forecast_with_prophet(overall_df, periods=12, frequency='W')
    
#     # Get latest transactions for the dashboard
#     latest_transactions = Transaction.objects.all().order_by('-timestamp')[:10]
    
#     # Get items that are predicted to be sold out soon
#     items_with_stockout = []
#     for item in items:
#         item_df = prepare_time_series_data(item_id=item.id, period='weekly')
#         if not item_df.empty:
#             forecast, _ = forecast_with_prophet(item_df, periods=12, frequency='W')
#             stockout_date = predict_stockout(item.id, forecast)
#             if stockout_date:
#                 # Calculate days until stockout
#                 today = timezone.now().date()
#                 days_until = (stockout_date.date() - today).days if hasattr(stockout_date, 'date') else None
                
#                 items_with_stockout.append({
#                     'item': item,
#                     'stockout_date': stockout_date,
#                     'days_until': days_until
#                 })
    
#     # Sort by stockout date
#     if items_with_stockout:
#         items_with_stockout.sort(key=lambda x: x['stockout_date'])
    
#     context = {
#         'overall_chart': overall_chart,
#         'items_with_stockout': items_with_stockout,
#         'latest_transactions': latest_transactions,
#         'items': items,  # Add all items for the chart
#     }
    
#     return render(request, 'AApp/index.html', context)

# def calculate_forecast_accuracy(actual_df, forecast_df):
#     """
#     Calculate various accuracy metrics for the forecast
#     - MAPE (Mean Absolute Percentage Error)
#     - RMSE (Root Mean Square Error)
#     - MAE (Mean Absolute Error)
#     """
#     if actual_df.empty or forecast_df.empty:
#         return {
#             'mape': None,
#             'rmse': None,
#             'mae': None
#         }
    
#     # Merge actual and forecasted data
#     merged_df = pd.merge(
#         actual_df, 
#         forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
#         on='ds', 
#         how='inner'
#     )
    
#     if merged_df.empty:
#         return {
#             'mape': None,
#             'rmse': None,
#             'mae': None
#         }
    
#     # Calculate errors
#     merged_df['error'] = merged_df['y'] - merged_df['yhat']
#     merged_df['abs_error'] = abs(merged_df['error'])
#     merged_df['squared_error'] = merged_df['error'] ** 2
    
#     # Calculate MAPE (avoiding division by zero)
#     merged_df['abs_percent_error'] = np.where(
#         merged_df['y'] != 0,
#         100.0 * merged_df['abs_error'] / merged_df['y'],
#         np.nan
#     )
    
#     # Calculate metrics
#     mape = merged_df['abs_percent_error'].dropna().mean()
#     rmse = np.sqrt(merged_df['squared_error'].mean())
#     mae = merged_df['abs_error'].mean()
    
#     return {
#         'mape': mape,
#         'rmse': rmse,
#         'mae': mae,
#         'merged_df': merged_df  # Return the merged data for additional analysis
#     }

# def forecast_with_prophet(df, periods=12, frequency='W'):
#     # Check if DataFrame is empty or has insufficient data
#     if df.empty or len(df) < 2:  # Prophet needs at least 2 data points
#         # Create a blank image for the chart
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.text(0.5, 0.5, 'Insufficient data for forecast', 
#                 horizontalalignment='center', verticalalignment='center',
#                 transform=ax.transAxes, fontsize=14)
#         plt.close(fig)
        
#         # Convert plot to base64 for embedding in HTML
#         buffer = io.BytesIO()
#         fig.savefig(buffer, format='png')
#         buffer.seek(0)
#         image_png = buffer.getvalue()
#         buffer.close()
        
#         graphic = base64.b64encode(image_png).decode('utf-8')
        
#         # Return empty forecast and the "no data" image
#         return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper']), graphic
    
#     # Split data into training and test sets (80/20)
#     train_size = int(len(df) * 0.8)
#     train_df = df.iloc[:train_size].copy()
#     test_df = df.iloc[train_size:].copy() if train_size < len(df) else pd.DataFrame()
    
#     # Initialize and train Prophet model
#     model = Prophet(
#         yearly_seasonality=True,
#         weekly_seasonality=True,
#         daily_seasonality=False,
#         changepoint_prior_scale=0.05
#     )
#     model.fit(train_df)
    
#     # Create future dataframe for prediction
#     # Include both test period and future periods
#     if test_df.empty:
#         future = model.make_future_dataframe(periods=periods, freq=frequency)
#     else:
#         # Make sure we're forecasting over the test period plus additional periods
#         test_periods = len(test_df)
#         future = model.make_future_dataframe(periods=test_periods + periods, freq=frequency)
    
#     # Generate forecast
#     forecast = model.predict(future)
    
#     # Calculate accuracy metrics if we have test data
#     accuracy_metrics = None
#     if not test_df.empty:
#         accuracy_metrics = calculate_forecast_accuracy(test_df, forecast)
    
#     # Create visualization with metrics
#     fig = model.plot(forecast)
#     ax = fig.gca()
    
#     # Add metrics to the plot if available
#     if accuracy_metrics and accuracy_metrics['mape'] is not None:
#         metrics_text = (
#             f"MAPE: {accuracy_metrics['mape']:.2f}%\n"
#             f"RMSE: {accuracy_metrics['rmse']:.2f}\n"
#             f"MAE: {accuracy_metrics['mae']:.2f}"
#         )
#         plt.figtext(0.01, 0.01, metrics_text, fontsize=10)
    
#     plt.title('Sales Forecast with Accuracy Metrics')
    
#     # Convert plot to base64 for embedding in HTML
#     buffer = io.BytesIO()
#     fig.savefig(buffer, format='png')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     buffer.close()
    
#     graphic = base64.b64encode(image_png).decode('utf-8')
    
#     return forecast, graphic, accuracy_metrics


def prepare_time_series_data(item_id=None, period='weekly'):
    # Query transactions, filter by item if specified
    queryset = Transaction.objects.all()
    if item_id:
        queryset = queryset.filter(item_id=item_id)
    
    # Group by time period and sum quantities
    if period == 'weekly':
        # Extract week from timestamp and group
        data = queryset.annotate(
            week=models.functions.TruncWeek('timestamp')
        ).values('week').annotate(
            total_quantity=Sum('quantity')
        ).order_by('week')
        
        # Convert to DataFrame
        df = pd.DataFrame(list(data))
        if df.empty:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['ds', 'y'])
            
        df.rename(columns={'week': 'ds', 'total_quantity': 'y'}, inplace=True)
        
        # Remove timezone information
        df['ds'] = df['ds'].dt.tz_localize(None)
    
    elif period == 'monthly':
        # Extract month from timestamp and group
        data = queryset.annotate(
            month=models.functions.TruncMonth('timestamp')
        ).values('month').annotate(
            total_quantity=Sum('quantity')
        ).order_by('month')
        
        # Convert to DataFrame
        df = pd.DataFrame(list(data))
        if df.empty:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['ds', 'y'])
            
        df.rename(columns={'month': 'ds', 'total_quantity': 'y'}, inplace=True)
        
        # Remove timezone information
        df['ds'] = df['ds'].dt.tz_localize(None)
    
    return df

def calculate_forecast_accuracy(actual_df, forecast_df):
    """
    Calculate various accuracy metrics for the forecast
    - MAPE (Mean Absolute Percentage Error)
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    """
    if actual_df.empty or forecast_df.empty:
        return {
            'mape': None,
            'rmse': None,
            'mae': None
        }
    
    # Merge actual and forecasted data
    merged_df = pd.merge(
        actual_df, 
        forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='inner'
    )
    
    if merged_df.empty:
        return {
            'mape': None,
            'rmse': None,
            'mae': None
        }
    
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
    mape = float(merged_df['abs_percent_error'].dropna().mean())
    rmse = float(np.sqrt(merged_df['squared_error'].mean()))
    mae = float(merged_df['abs_error'].mean())
    
    # Return metrics as plain Python types, not numpy types
    return {
        'mape': mape,
        'rmse': rmse,
        'mae': mae
    }

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


def forecast_with_prophet(df, periods=12, frequency='W'):
    # Check if DataFrame is empty or has insufficient data
    if df.empty or len(df) < 2:  # Prophet needs at least 2 data points
        # Create a blank image for the chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Insufficient data for forecast', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        plt.close(fig)
        
        # Convert plot to base64 for embedding in HTML
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        
        # Return empty forecast and the "no data" image
        return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper']), graphic, None
    
    # Split data into training and test sets (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy() if train_size < len(df) else pd.DataFrame()
    
    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(train_df)
    
    # Create future dataframe for prediction
    # Include both test period and future periods
    if test_df.empty:
        future = model.make_future_dataframe(periods=periods, freq=frequency)
    else:
        # Make sure we're forecasting over the test period plus additional periods
        test_periods = len(test_df)
        future = model.make_future_dataframe(periods=test_periods + periods, freq=frequency)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Calculate accuracy metrics if we have test data
    accuracy_metrics = None
    if not test_df.empty:
        accuracy_metrics = calculate_forecast_accuracy(test_df, forecast)
    
    # Extract just the numeric metrics for template rendering
    template_metrics = None
    if accuracy_metrics and accuracy_metrics.get('mape') is not None:
        template_metrics = {
            'mape': float(accuracy_metrics['mape']) if accuracy_metrics['mape'] is not None else None,
            'rmse': float(accuracy_metrics['rmse']) if accuracy_metrics['rmse'] is not None else None,
            'mae': float(accuracy_metrics['mae']) if accuracy_metrics['mae'] is not None else None
        }
    
    # Create visualization with metrics
    fig = model.plot(forecast)
    ax = fig.gca()
    
    # Add metrics to the plot if available
    if template_metrics:
        metrics_text = (
            f"MAPE: {template_metrics['mape']:.2f}%\n"
            f"RMSE: {template_metrics['rmse']:.2f}\n"
            f"MAE: {template_metrics['mae']:.2f}"
        )
        plt.figtext(0.01, 0.01, metrics_text, fontsize=10)
    
    plt.title('Sales Forecast with Accuracy Metrics')
    
    # Convert plot to base64 for embedding in HTML
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    return forecast, graphic, template_metrics


def predict_stockout(item_id, forecast_df):
    # Check if forecast is empty
    if forecast_df.empty:
        return None
        
    # Get current stock level
    try:
        item = Item.objects.get(id=item_id)
        current_stock = item.quantity
        
        # Extract the forecasted values
        forecast_values = forecast_df[['ds', 'yhat']].copy()
        
        # Calculate cumulative sales
        forecast_values['cumulative_sales'] = forecast_values['yhat'].cumsum()
        
        # Find when cumulative sales exceeds current stock
        stockout_date = None
        for index, row in forecast_values.iterrows():
            if row['cumulative_sales'] >= current_stock:
                stockout_date = row['ds']
                break
        
        return stockout_date
    except Item.DoesNotExist:
        return None

def item_analytics_view(request, item_id):
    # Get the item
    item = Item.objects.get(id=item_id)
    
    # Prepare weekly data
    weekly_df = prepare_time_series_data(item_id=item_id, period='weekly')
    monthly_df = prepare_time_series_data(item_id=item_id, period='monthly')
    
    # Generate forecasts with accuracy metrics
    weekly_forecast, weekly_chart, weekly_metrics = forecast_with_prophet(weekly_df, periods=12, frequency='W')
    monthly_forecast, monthly_chart, monthly_metrics = forecast_with_prophet(monthly_df, periods=6, frequency='M')
    
    # Calculate performance metrics
    performance_metrics = calculate_item_performance_metrics(item_id=item_id)
    
    # Predict stockout date
    stockout_date = predict_stockout(item_id, weekly_forecast)
    
    context = {
        'item': item,
        'weekly_chart': weekly_chart,
        'monthly_chart': monthly_chart,
        'stockout_date': stockout_date,
        'weekly_metrics': weekly_metrics,
        'monthly_metrics': monthly_metrics,
        'performance_metrics': performance_metrics,
    }
    
    return render(request, 'AApp/analytics/item_analytics.html', context)

def analytics_dashboard(request):
    # Get all items
    items = Item.objects.all()
    
    # Get overall sales trend
    overall_df = prepare_time_series_data(period='weekly')
    overall_forecast, overall_chart, overall_metrics = forecast_with_prophet(overall_df, periods=12, frequency='W')
    
    # Calculate overall performance metrics
    overall_performance = calculate_item_performance_metrics(period='monthly')
    
    # Get latest transactions for the dashboard
    latest_transactions = Transaction.objects.all().order_by('-timestamp')[:10]
    
    # Get items that are predicted to be sold out soon
    items_with_stockout = []
    items_performance = []
    
    for item in items:
        item_df = prepare_time_series_data(item_id=item.id, period='weekly')
        
        # Get item performance metrics 
        item_metrics = calculate_item_performance_metrics(item_id=item.id)
        
        # Add performance metrics to list
        items_performance.append({
            'item': item,
            'metrics': item_metrics
        })
        
        if not item_df.empty:
            forecast, _, _ = forecast_with_prophet(item_df, periods=12, frequency='W')
            stockout_date = predict_stockout(item.id, forecast)
            if stockout_date:
                # Calculate days until stockout
                today = timezone.now().date()
                days_until = (stockout_date.date() - today).days if hasattr(stockout_date, 'date') else None
                
                items_with_stockout.append({
                    'item': item,
                    'stockout_date': stockout_date,
                    'days_until': days_until
                })
    
    # Sort by stockout date
    if items_with_stockout:
        items_with_stockout.sort(key=lambda x: x['stockout_date'])
    
    # Sort items by inventory turnover (highest first)
    items_performance.sort(
        key=lambda x: x['metrics']['inventory_turnover'] if x['metrics']['inventory_turnover'] is not None else 0, 
        reverse=True
    )
    
    context = {
        'overall_chart': overall_chart,
        'overall_metrics': overall_metrics,
        'overall_performance': overall_performance,
        'items_with_stockout': items_with_stockout,
        'items_performance': items_performance,
        'latest_transactions': latest_transactions,
        'items': items,  # Add all items for the chart
    }
    
    return render(request, 'AApp/index.html', context)

















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




