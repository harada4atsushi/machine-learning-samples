# -*- coding: utf-8 -
from tinydb import TinyDB, Query

db = TinyDB('db/dataset.json')
db.purge()
db.insert({'day_of_week': 1, 'attend': 0})
db.insert({'day_of_week': 1, 'attend': 1})
db.insert({'day_of_week': 1, 'attend': 0})
db.insert({'day_of_week': 2, 'attend': 1})
db.insert({'day_of_week': 2, 'attend': 1})
db.insert({'day_of_week': 2, 'attend': 0})
db.insert({'day_of_week': 3, 'attend': 1})
db.insert({'day_of_week': 4, 'attend': 1})
db.insert({'day_of_week': 5, 'attend': 0})
db.insert({'day_of_week': 5, 'attend': 0})
db.insert({'day_of_week': 6, 'attend': 1})
db.insert({'day_of_week': 7, 'attend': 1})
