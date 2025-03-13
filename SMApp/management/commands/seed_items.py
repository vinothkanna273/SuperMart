from django.core.management.base import BaseCommand
from SMApp.models import Item


class Command(BaseCommand):
    help = 'Seed the database with basic supermarket items'

    def handle(self, *args, **kwargs):
        # items = [
        #     {"name": "Rice (5kg)", "description": "Premium quality rice, 5kg pack",
        #      "quantity": 100, "price": 250.00},
        #     {"name": "Wheat Flour (2kg)", "description": "Freshly milled whole wheat flour, 2kg pack",
        #      "quantity": 50, "price": 90.00},
        #     {"name": "Sugar (1kg)", "description": "Refined sugar, 1kg pack",
        #      "quantity": 75, "price": 45.00},
        #     {"name": "Salt (1kg)", "description": "Iodized table salt, 1kg pack",
        #      "quantity": 150, "price": 20.00},
        #     {"name": "Cooking Oil (1L)", "description": "Sunflower cooking oil, 1L bottle",
        #      "quantity": 80, "price": 160.00},
        #     {"name": "Milk (1L)", "description": "Full cream milk, 1L packet",
        #      "quantity": 120, "price": 60.00},
        #     {"name": "Eggs (12 pcs)", "description": "Fresh chicken eggs, pack of 12",
        #      "quantity": 30, "price": 70.00},
        #     {"name": "Bread (Loaf)", "description": "Whole wheat bread loaf, 400g",
        #      "quantity": 40, "price": 50.00},
        #     {"name": "Butter (500g)", "description": "Creamy salted butter, 500g pack",
        #      "quantity": 25, "price": 200.00},
        #     {"name": "Cheese (200g)", "description": "Processed cheese, 200g pack",
        #      "quantity": 20, "price": 100.00},
        #     {"name": "Biscuits (200g)", "description": "Assorted cream biscuits, 200g pack",
        #      "quantity": 60, "price": 30.00},
        #     {"name": "Instant Noodles (75g)", "description": "Spicy instant noodles, 75g pack",
        #      "quantity": 100, "price": 12.00},
        #     {"name": "Tea (250g)", "description": "Premium loose leaf tea, 250g pack",
        #      "quantity": 40, "price": 150.00},
        #     {"name": "Coffee (100g)", "description": "Instant coffee powder, 100g jar",
        #      "quantity": 35, "price": 180.00},
        #     {"name": "Shampoo (200ml)", "description": "Herbal shampoo, 200ml bottle",
        #      "quantity": 50, "price": 120.00},
        #     {"name": "Soap (100g)", "description": "Moisturizing bath soap, 100g bar",
        #      "quantity": 90, "price": 40.00},
        #     {"name": "Toothpaste (200g)", "description": "Mint-flavored toothpaste, 200g tube",
        #      "quantity": 60, "price": 80.00},
        #     {"name": "Toilet Cleaner (500ml)", "description": "Powerful toilet cleaner, 500ml bottle",
        #      "quantity": 25, "price": 90.00},
        #     {"name": "Dishwashing Liquid (1L)", "description": "Lemon-scented dishwashing liquid, 1L bottle",
        #      "quantity": 30, "price": 150.00},
        #     {"name": "Vegetables (Assorted)", "description": "Fresh seasonal vegetables, 1kg",
        #      "quantity": 50, "price": 60.00},
        #     {"name": "Fruits (Assorted)", "description": "Fresh seasonal fruits, 1kg",
        #      "quantity": 40, "price": 100.00},
        #     {"name": "Potatoes (1kg)", "description": "Fresh potatoes, 1kg pack",
        #      "quantity": 80, "price": 25.00},
        #     {"name": "Onions (1kg)", "description": "Fresh onions, 1kg pack",
        #      "quantity": 80, "price": 30.00},
        #     {"name": "Tomatoes (1kg)", "description": "Ripe red tomatoes, 1kg pack",
        #      "quantity": 80, "price": 35.00},
        #     {"name": "Bananas (1 dozen)", "description": "Fresh bananas, dozen pack",
        #      "quantity": 40, "price": 50.00}
        # ]

        items = []

        for item in items:
            Item.objects.get_or_create(
                name=item["name"],
                defaults={
                    "description": item["description"],
                    "quantity": item["quantity"],
                    "price": item["price"],
                },
            )

        self.stdout.write(self.style.SUCCESS('Successfully seeded items!'))

