from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

from app.views import main
app.register_blueprint(main.bp)
