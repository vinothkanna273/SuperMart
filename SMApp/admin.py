from django.contrib import admin
from .models import Item, Floor, Shelf, Transaction


admin.site.site_header='SuperMart admin Dashboard'

@admin.register(Item)
class ItemAdmin(admin.ModelAdmin):
    list_display = ('name', 'quantity', 'price', 'added_on')
    search_fields = ('name',)
    list_filter = ('added_on',)

# admin.site.register(Shelf)
@admin.register(Shelf)
class ShelfAdmin(admin.ModelAdmin):
    list_display = ('floor', 'cell_index', 'item_ids', 'item_names', 'mode')
    search_fields = ('item_names',)

admin.site.register(Floor)

# admin.site.register(Transaction)
@admin.register(Transaction)
class ShelfAdmin(admin.ModelAdmin):
    list_display = ('item_name', 'user', 'timestamp', 'quantity', 'total_price')
    search_fields = ('item_name',)
