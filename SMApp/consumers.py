import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from django.core.cache import cache

class CartConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Join cart group
        await self.channel_layer.group_add(
            'cart_updates',
            self.channel_name
        )
        await self.accept()
        
        # Send initial data
        detected_items = await sync_to_async(cache.get)('detected_items', [])
        matched_items = await sync_to_async(cache.get)('detected_matched_items', [])
        
        await self.send(text_data=json.dumps({
            'type': 'cart_update',
            'detected_items': detected_items,
            'matched_items': matched_items
        }))

    async def disconnect(self, close_code):
        # Leave cart group
        await self.channel_layer.group_discard(
            'cart_updates',
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        # If client sends data, handle it here
        pass

    # Receive message from cart group
    async def cart_update(self, event):
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'detected_items': event['detected_items'],
            'matched_items': event['matched_items']
        }))