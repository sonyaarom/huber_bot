from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

from app.views.main import bp as main_bp
app.register_blueprint(main_bp)
