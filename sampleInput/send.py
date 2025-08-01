import requests
import json

# Function to get the local IP address of your Django server
def get_django_ip():
    return "192.168.168.26"  # Replace with your actual local IP

# Function to get the port your Django app is running on
def get_django_port():
    return 8000  # Default Django dev server port

# Function to send cart updates to Django
def send_cart_update(cart_data):
    django_ip = get_django_ip()
    django_port = get_django_port()
    
    url = f"http://{django_ip}:{django_port}/api/update-cart/"
    
    payload = {
        "cart_id": "raspberry_pi_cart",
        "items": cart_data.get("items", {}),
        "total_cost": cart_data.get("total_cost", 0.0),
        "total_weight": cart_data.get("total_weight", 0.0)
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("Cart data successfully sent to Django app")
        else:
            print(f"Failed to send cart data. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error sending cart data: {e}")

# Cart data options
cart_full = {
    "items": {
        "kodomos": {
            "quantity": 1, "price": 30, "total_price": 30, "confidence": 0.9465694427490234, "weight": 4.564365833463171
        },
        "oponds": {
            "quantity": 1, "price": 70, "total_price": 70, "confidence": 0.963674008846283, "weight": 29.66868027348778
        }
    },
    "total_cost": 100.0,
    "total_weight": 34.23304610695095
}

cart_empty = {
    "items": {},
    "total_cost": 0.0,
    "total_weight": 0.0
}

cart_partial = {
    "items": {
	"ndettol": {
            "quantity": 1, "price": 100, "total_price": 100, "confidence": 0.879852547, "weight": 150.0145
        }

    },
    "total_cost": 100.0,
    "total_weight": 150.0145
}

# User input for cart selection
print("Enter choice:")
print("1 - Send full cart data")
print("2 - Send empty cart data")
print("3 - Send partial cart data")
while(True):
    choice = input("Your choice: ")

    if choice == "1":
        send_cart_update(cart_full)
    elif choice == "2":
        send_cart_update(cart_empty)
    elif choice == "3":
        send_cart_update(cart_partial)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
