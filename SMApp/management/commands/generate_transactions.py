# Save this file as management/commands/generate_transactions.py in your Django app

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.contrib.auth.models import User
from SMApp.models import Item, Transaction
import random
import datetime

class Command(BaseCommand):
    help = 'Generates dummy transaction data with realistic patterns'

    def handle(self, *args, **kwargs):
        # Constants
        USER_ID = 1
        START_DATE = timezone.make_aware(datetime.datetime(2025, 1, 1))  # Starting from January 1, 2025
        NUM_DAYS = 90  # Generate 3 months of data

        # Get all items from the database
        items = Item.objects.all()
        user = User.objects.get(id=USER_ID)
        
        # Item purchase patterns (item_id, typical_quantity, weekend_multiplier, is_weekly, is_biweekly, is_monthly)
        item_patterns = {
            # Daily essentials - bought frequently
            "Milk (1L)": {"typical_qty": 2, "weekend_mult": 1.5, "is_weekly": False, "is_biweekly": False, "is_monthly": False},
            "Bread (Loaf)": {"typical_qty": 1, "weekend_mult": 1.5, "is_weekly": False, "is_biweekly": False, "is_monthly": False},
            "Eggs (12 pcs)": {"typical_qty": 1, "weekend_mult": 1.2, "is_weekly": False, "is_biweekly": False, "is_monthly": False},
            
            # Weekly purchases - regular pattern
            "Vegetables (Assorted)": {"typical_qty": 1, "weekend_mult": 2.0, "is_weekly": True, "is_biweekly": False, "is_monthly": False},
            "Fruits (Assorted)": {"typical_qty": 1, "weekend_mult": 1.8, "is_weekly": True, "is_biweekly": False, "is_monthly": False},
            
            # Biweekly purchases
            "Rice (5kg)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": True, "is_monthly": False},
            "Wheat Flour (2kg)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": True, "is_monthly": False},
            "Sugar (1kg)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": True, "is_monthly": False},
            "Cooking Oil (1L)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": True, "is_monthly": False},
            "Tea (250g)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": True, "is_monthly": False},
            "Coffee (100g)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": True, "is_monthly": False},
            
            # Monthly purchases
            "Shampoo (200ml)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": False, "is_monthly": True},
            "Toothpaste (200g)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": False, "is_monthly": True},
            "Dishwashing Liquid (1L)": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": False, "is_monthly": True},
            "Dettol": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": False, "is_monthly": True},
            "Vaseline": {"typical_qty": 1, "weekend_mult": 1, "is_weekly": False, "is_biweekly": False, "is_monthly": True},
        }
        
        # Add all other items as irregular purchases
        for item in items:
            item_name = item.name
            if item_name not in item_patterns:
                item_patterns[item_name] = {
                    "typical_qty": 1, 
                    "weekend_mult": 1, 
                    "is_weekly": False, 
                    "is_biweekly": False, 
                    "is_monthly": False
                }
                
                # Some items have specific patterns
                if "Biscuits" in item_name:
                    item_patterns[item_name]["typical_qty"] = 2
                    item_patterns[item_name]["weekend_mult"] = 1.5
                elif "Noodles" in item_name:
                    item_patterns[item_name]["typical_qty"] = 3
                    item_patterns[item_name]["weekend_mult"] = 1.2
                elif "Cheese" in item_name:
                    item_patterns[item_name]["weekend_mult"] = 1.3
                elif "Potatoes" in item_name or "Onions" in item_name or "Tomatoes" in item_name:
                    item_patterns[item_name]["typical_qty"] = 2

        # Track transactions created
        transactions_created = 0
        
        # For each day in our range
        for day in range(NUM_DAYS):
            current_date = START_DATE + datetime.timedelta(days=day)
            is_weekend = current_date.weekday() >= 5  # 5 is Saturday, 6 is Sunday
            day_of_month = current_date.day
            
            # Process each item
            for item in items:
                item_name = item.name
                pattern = item_patterns.get(item_name, {
                    "typical_qty": 1, 
                    "weekend_mult": 1, 
                    "is_weekly": False, 
                    "is_biweekly": False, 
                    "is_monthly": False
                })
                
                buy_today = False
                quantity = pattern["typical_qty"]
                
                # Apply purchase patterns
                if pattern["is_weekly"] and day % 7 == 5:  # Buy weekly items on Saturdays
                    buy_today = True
                elif pattern["is_biweekly"] and day % 14 == 0:  # Buy biweekly items every two weeks
                    buy_today = True
                elif pattern["is_monthly"] and day_of_month == 1:  # Buy monthly items on 1st of month
                    buy_today = True
                elif not pattern["is_weekly"] and not pattern["is_biweekly"] and not pattern["is_monthly"]:
                    # For daily and irregular items, use probability
                    if "Milk" in item_name or "Bread" in item_name or "Eggs" in item_name:
                        # Daily essentials have high purchase probability
                        buy_today = random.random() < 0.7
                    else:
                        # Other irregular items have lower purchase probability
                        buy_today = random.random() < 0.2
                
                # Adjust quantity for weekends if applicable
                if is_weekend and buy_today:
                    quantity = int(pattern["typical_qty"] * pattern["weekend_mult"])
                
                # Add some randomness to quantity
                if buy_today:
                    quantity_variation = random.choice([0, 0, 0, 1, -1])  # More likely to be the typical quantity
                    quantity = max(1, quantity + quantity_variation)
                
                # If we're buying this item today, add a transaction
                if buy_today:
                    # Calculate total price
                    total_price = item.price * quantity
                    
                    # Add a random hour during the day
                    hour = random.randint(9, 20)
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    transaction_time = current_date.replace(hour=hour, minute=minute, second=second)
                    
                    # Create the transaction
                    transaction = Transaction(
                        item=item,
                        item_name=item.name,
                        quantity=quantity,
                        total_price=total_price,
                        user=user,
                        timestamp=transaction_time
                    )
                    transaction.save()
                    transactions_created += 1
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created {transactions_created} transactions over {NUM_DAYS} days'))