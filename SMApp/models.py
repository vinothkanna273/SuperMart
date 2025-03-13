from django.db import models


class Item(models.Model):
    name = models.CharField(max_length=100, unique=True)  
    description = models.TextField(blank=True, null=True)  
    quantity = models.PositiveIntegerField(default=1)  
    price = models.DecimalField(max_digits=10, decimal_places=2)  
    added_on = models.DateTimeField(auto_now_add=True)  

    def __str__(self):
        return f'{self.name} - {self.quantity} - ₹{self.price}'
    
class Floor(models.Model):
    name = models.CharField(max_length=100, unique=True)
    length = models.IntegerField()
    width = models.IntegerField()
    data = models.JSONField(default=dict) #stores the shelves data.

    def __str__(self):
        return f'{self.name} - Len: {self.length} - Wid: {self.width}'


class Shelf(models.Model):
    floor = models.ForeignKey(Floor, on_delete=models.CASCADE)
    cell_index = models.IntegerField()
    item_ids = models.JSONField(default=list)
    item_names = models.JSONField(default=list)
    mode = models.CharField(max_length=20)
