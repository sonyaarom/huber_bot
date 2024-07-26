from flask import Blueprint, render_template
from flask_socketio import emit
from app import socketio

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(msg):
    response = process_message(msg)
    emit('message', response, broadcast=True)

def process_message(msg):
    # Примитивная логика ответа
    if 'hello' in msg.lower():
        return "Bot: Hello! How can I help you today?"
    elif 'how are you' in msg.lower():
        return "Bot: I'm just a bot, but I'm doing great! How about you?"
    else:
        return f"Bot: You said '{msg}'"
