# Generated by Django 5.0.6 on 2025-08-01 10:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SMApp', '0007_alter_transaction_timestamp'),
    ]

    operations = [
        migrations.AlterField(
            model_name='transaction',
            name='timestamp',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
