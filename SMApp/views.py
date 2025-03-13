from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse
from django.core.cache import cache
import json
import pandas as pd
import csv
import docx
import urllib.parse
from collections import deque


from .models import Item, Floor, Shelf

import logging
logger = logging.getLogger(__name__)

# Create your views here.


def index(request):
    # pull data from db, transform, send email
    # return HttpResponse('Hello World')
    return render(request, 'AApp/index.html')


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

# @csrf_exempt
# def customer_view_floor(request, floor_name):
#     try:
#         items = Item.objects.all()
#         floor = Floor.objects.get(name=floor_name)
#         shelves = Shelf.objects.filter(floor=floor).values(
#             'cell_index', 'mode', 'item_ids', 'item_names')
        
#         # Convert shelves to a dictionary format that matches your grid data
#         shelves_dict = {shelf['cell_index']: {
#             'mode': shelf['mode'],
#             'itemIds': shelf['item_ids'],
#             'itemNames': shelf['item_names']
#         } for shelf in shelves}
        
#         # Get uploaded items from session if available
#         uploaded_items = request.session.get('uploaded_items', [])
#         if uploaded_items:
#             # Clear from session after retrieving
#             del request.session['uploaded_items']
#             request.session.modified = True
        
#         context = {
#             'items': items,
#             'floor': floor, 
#             'shelves': list(shelves),
#             'floor_data_json': json.dumps(floor.data),
#             'uploaded_items_json': json.dumps(uploaded_items)
#         }
#         return render(request, 'AApp/customer/viewCustomerFloor.html', context)
#     except Floor.DoesNotExist:
#         return render(request, 'AApp/customer/viewCustomerFloor.html', {'floor_name': floor_name})


# @csrf_exempt
# def customer_view_floor(request, floor_name):
#     try:
#         items = Item.objects.all()
#         floor = Floor.objects.get(name=floor_name)
#         shelves = Shelf.objects.filter(floor=floor).values(
#             'cell_index', 'mode', 'item_ids', 'item_names')

#         uploaded_items = request.session.get('uploaded_items', [])

#         context = {
#             'items': items,
#             'floor': floor,
#             'shelves': list(shelves),
#             'floor_data_json': json.dumps(floor.data),
#             'uploaded_items_json': json.dumps(uploaded_items),
#             'uploaded_items': uploaded_items
#         }
#         return render(request, 'AApp/customer/viewCustomerFloor.html', context)
    
#     except Floor.DoesNotExist:
#         return render(request, 'AApp/customer/viewCustomerFloor.html', {'floor_name': floor_name})


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
#             detected_items = []  # Create a list to store detected item names
            
#             # Extract item names from the items dictionary
#             for item_name in items.keys():
#                 detected_items.append(item_name)
                
#                 db_items = Item.objects.filter(name__iexact=item_name)
                
#                 if not db_items.exists():
#                     modified_name = item_name[1:]
#                     db_items = Item.objects.filter(name__icontains=modified_name)
                
#                 if db_items.exists():
#                     for db_item in db_items:
#                         matched_items.append({
#                             "name": db_item.name,
#                             "quantity": db_item.quantity,
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
                
#             # Store the detected items in cache instead of session
#             cache.set('detected_items', detected_items, timeout=None)  # No timeout
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

@csrf_exempt
def update_cart(request):
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

            # Print matched items
            if matched_items:
                print("\nMatched Items in Database:")
                for item in matched_items:
                    print(f"Name: {item['name']}, Quantity: {item['quantity']}, Price: ₹{item['price']}, Description: {item['description']}")
            else:
                print("\nNo matching items found in database.")
                
            # Store detected items in cache
            cache.set('detected_items', detected_items, timeout=None)  
            cache.set('detected_matched_items', matched_items, timeout=None)
            
            print(f"Stored in cache - detected_items: {detected_items}")
            print(f"Stored in cache - matched items count: {len(matched_items)}")

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




