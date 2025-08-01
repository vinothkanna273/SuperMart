# Save this file as management/commands/generate_transactions.py in your Django app

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.contrib.auth.models import User
from SMApp.models import Item, Transaction
import random
import datetime
from decimal import Decimal

class Command(BaseCommand):
    help = 'Generates realistic transaction data for the past 7 months'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user-id',
            type=int,
            default=1,
            help='User ID to generate transactions for'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing transactions before generating new ones'
        )

    def handle(self, *args, **kwargs):
        # Configuration
        USER_ID = kwargs['user_id']
        CLEAR_EXISTING = kwargs['clear_existing']
        
        # Generate for past 7 months
        END_DATE = timezone.now()
        START_DATE = END_DATE - datetime.timedelta(days=210)  # ~7 months
        
        if CLEAR_EXISTING:
            Transaction.objects.filter(user_id=USER_ID).delete()
            self.stdout.write(self.style.WARNING('Cleared existing transactions'))

        # Get user and items
        try:
            user = User.objects.get(id=USER_ID)
            items = list(Item.objects.all())
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'User with ID {USER_ID} does not exist'))
            return

        if not items:
            self.stdout.write(self.style.ERROR('No items found in database'))
            return

        # Enhanced item patterns with seasonal variations
        item_patterns = {
            # Daily essentials - high frequency
            "Milk (1L)": {
                "frequency": "daily", 
                "base_probability": 0.9,  # Increased
                "typical_qty": (1, 3), 
                "weekend_boost": 1.3,
                "seasonal_boost": {"winter": 1.1, "summer": 0.9}
            },
            "Bread (Loaf)": {
                "frequency": "daily", 
                "base_probability": 0.8,  # Increased
                "typical_qty": (1, 2), 
                "weekend_boost": 1.4,
                "seasonal_boost": {"winter": 1.0, "summer": 1.0}
            },
            "Eggs (12 pcs)": {
                "frequency": "daily", 
                "base_probability": 0.7,  # Increased
                "typical_qty": (1, 2), 
                "weekend_boost": 1.2,
                "seasonal_boost": {"winter": 1.1, "summer": 1.0}
            },
            
            # Weekly purchases
            "Vegetables (Assorted)": {
                "frequency": "weekly", 
                "base_probability": 0.9,  # Increased
                "typical_qty": (1, 3), 
                "weekend_boost": 1.8,
                "seasonal_boost": {"winter": 0.8, "summer": 1.2}
            },
            "Fruits (Assorted)": {
                "frequency": "weekly", 
                "base_probability": 0.85,  # Increased
                "typical_qty": (1, 2), 
                "weekend_boost": 1.6,
                "seasonal_boost": {"winter": 0.7, "summer": 1.3}
            },
            "Chicken (1kg)": {
                "frequency": "weekly", 
                "base_probability": 0.75,  # Increased
                "typical_qty": (1, 2), 
                "weekend_boost": 1.5,
                "seasonal_boost": {"winter": 1.1, "summer": 1.0}
            },
            
            # Bi-weekly staples
            "Rice (5kg)": {
                "frequency": "biweekly", 
                "base_probability": 0.9, 
                "typical_qty": (1, 1), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.0, "summer": 1.0}
            },
            "Wheat Flour (2kg)": {
                "frequency": "biweekly", 
                "base_probability": 0.8, 
                "typical_qty": (1, 2), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.2, "summer": 0.9}
            },
            "Sugar (1kg)": {
                "frequency": "biweekly", 
                "base_probability": 0.7, 
                "typical_qty": (1, 1), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.1, "summer": 1.0}
            },
            "Cooking Oil (1L)": {
                "frequency": "biweekly", 
                "base_probability": 0.8, 
                "typical_qty": (1, 2), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.0, "summer": 1.0}
            },
            "Tea (250g)": {
                "frequency": "monthly", 
                "base_probability": 0.9, 
                "typical_qty": (1, 2), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.3, "summer": 0.8}
            },
            "Coffee (100g)": {
                "frequency": "monthly", 
                "base_probability": 0.7, 
                "typical_qty": (1, 1), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.2, "summer": 1.0}
            },
            
            # Monthly household items
            "Shampoo (200ml)": {
                "frequency": "monthly", 
                "base_probability": 0.95, 
                "typical_qty": (1, 1), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.0, "summer": 1.0}
            },
            "Toothpaste (200g)": {
                "frequency": "monthly", 
                "base_probability": 0.9, 
                "typical_qty": (1, 2), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.0, "summer": 1.0}
            },
            "Dishwashing Liquid (1L)": {
                "frequency": "monthly", 
                "base_probability": 0.85, 
                "typical_qty": (1, 1), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.0, "summer": 1.0}
            },
            "Dettol": {
                "frequency": "monthly", 
                "base_probability": 0.8, 
                "typical_qty": (1, 1), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.2, "summer": 0.9}
            },
            "Vaseline": {
                "frequency": "monthly", 
                "base_probability": 0.6, 
                "typical_qty": (1, 1), 
                "weekend_boost": 1.0,
                "seasonal_boost": {"winter": 1.3, "summer": 0.8}
            },
        }

        def get_season(date):
            month = date.month
            if month in [12, 1, 2]:
                return "winter"
            elif month in [3, 4, 5]:
                return "spring"
            elif month in [6, 7, 8]:
                return "summer"
            else:
                return "autumn"

        def get_item_pattern(item_name):
            """Get pattern for item, with defaults for unknown items"""
            if item_name in item_patterns:
                return item_patterns[item_name]
            
            # Assign patterns based on item name keywords
            name_lower = item_name.lower()
            
            if any(keyword in name_lower for keyword in ['biscuit', 'snack', 'chocolate', 'candy']):
                return {
                    "frequency": "weekly", 
                    "base_probability": 0.6,  # Increased
                    "typical_qty": (1, 3), 
                    "weekend_boost": 1.8,
                    "seasonal_boost": {"winter": 1.2, "summer": 1.0}
                }
            elif any(keyword in name_lower for keyword in ['noodles', 'pasta', 'instant']):
                return {
                    "frequency": "weekly", 
                    "base_probability": 0.7,  # Increased
                    "typical_qty": (2, 4), 
                    "weekend_boost": 1.3,
                    "seasonal_boost": {"winter": 1.1, "summer": 0.9}
                }
            elif any(keyword in name_lower for keyword in ['cheese', 'butter', 'yogurt']):
                return {
                    "frequency": "weekly", 
                    "base_probability": 0.75,  # Increased
                    "typical_qty": (1, 2), 
                    "weekend_boost": 1.4,
                    "seasonal_boost": {"winter": 1.0, "summer": 0.9}
                }
            elif any(keyword in name_lower for keyword in ['soap', 'detergent', 'cleaner']):
                return {
                    "frequency": "monthly", 
                    "base_probability": 0.85,  # Increased
                    "typical_qty": (1, 1), 
                    "weekend_boost": 1.0,
                    "seasonal_boost": {"winter": 1.0, "summer": 1.0}
                }
            else:
                # Default pattern for unknown items
                return {
                    "frequency": "irregular", 
                    "base_probability": 0.6,  # Increased further
                    "typical_qty": (1, 2), 
                    "weekend_boost": 1.2,
                    "seasonal_boost": {"winter": 1.0, "summer": 1.0}
                }

        # Track last purchase dates for better patterns
        last_purchase = {}
        transactions_created = 0
        
        # Generate transactions day by day
        current_date = START_DATE
        while current_date <= END_DATE:
            is_weekend = current_date.weekday() >= 5
            is_payday = current_date.day in [1, 15]  # Assume payday on 1st and 15th
            season = get_season(current_date)
            
            # Determine daily shopping probability (higher on weekends and paydays)
            shopping_probability = 0.6  # Increased further
            if is_weekend:
                shopping_probability = 0.85  # Higher weekend probability
            if is_payday:
                shopping_probability += 0.15  # Payday boost
            
            # Decide if shopping happens today
            if random.random() < shopping_probability:
                daily_transactions = 0
                transaction_time_base = current_date.replace(
                    hour=random.randint(9, 20),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )
                
                for item in items:
                    pattern = get_item_pattern(item.name)
                    
                    # Calculate purchase probability
                    probability = pattern["base_probability"]
                    
                    # Apply weekend boost
                    if is_weekend:
                        probability *= pattern["weekend_boost"]
                    
                    # Apply seasonal boost
                    seasonal_multiplier = pattern["seasonal_boost"].get(season, 1.0)
                    probability *= seasonal_multiplier
                    
                    # Check frequency-based constraints
                    days_since_last = 999
                    if item.name in last_purchase:
                        days_since_last = (current_date.date() - last_purchase[item.name]).days
                    
                    frequency_ok = True
                    if pattern["frequency"] == "daily" and days_since_last < 1:
                        frequency_ok = False
                    elif pattern["frequency"] == "weekly" and days_since_last < 5:
                        frequency_ok = False
                    elif pattern["frequency"] == "biweekly" and days_since_last < 12:
                        frequency_ok = False
                    elif pattern["frequency"] == "monthly" and days_since_last < 25:
                        frequency_ok = False
                    
                    # Decide if we buy this item
                    if frequency_ok and random.random() < min(probability, 0.95):
                        # Determine quantity
                        min_qty, max_qty = pattern["typical_qty"]
                        quantity = random.randint(min_qty, max_qty)
                        
                        # Add small random variation
                        if random.random() < 0.2:  # 20% chance of quantity variation
                            quantity += random.choice([-1, 1])
                            quantity = max(1, quantity)
                        
                        # Calculate total price
                        total_price = item.price * quantity
                        
                        # Create transaction with slight time variation
                        transaction_time = transaction_time_base + datetime.timedelta(
                            minutes=random.randint(-30, 30)
                        )
                        
                        transaction = Transaction(
                            item=item,
                            item_name=item.name,
                            quantity=quantity,
                            total_price=total_price,
                            user=user,
                            timestamp=transaction_time
                        )
                        transaction.save()
                        
                        # Update last purchase date
                        last_purchase[item.name] = current_date.date()
                        transactions_created += 1
                        daily_transactions += 1
                        
                        # Limit transactions per day to make it realistic
                        if daily_transactions >= 20:  # Increased max items per shopping trip
                            break
            
            current_date += datetime.timedelta(days=1)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created {transactions_created} transactions over 7 months '
                f'({START_DATE.date()} to {END_DATE.date()})\n'
                f'Average: {transactions_created / 210:.1f} transactions per day'
            )
        )