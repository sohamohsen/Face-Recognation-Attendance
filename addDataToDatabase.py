import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase with credentials
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognation-528bd-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "42001": {
        "name": "Soha Mohsen",
    },
    "42002": {  # Change the key to ensure uniqueness
        "name": "Rawan Abo El elaa",
    }
}

for key, value in data.items():
    ref.child(key).set(value)
