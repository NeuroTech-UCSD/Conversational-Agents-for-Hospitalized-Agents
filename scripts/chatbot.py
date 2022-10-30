import time
import asyncio
import socketio
import random
from src.predict_utils import GUI
from src.settings import Configuration

sio = socketio.AsyncClient()
PORT = Configuration.app['port']
HOST = Configuration.app['host']


@sio.event
async def connect():
    print('chatbot connected')


@sio.event
async def connect_error(e):
    print('Connection error:', e)


@sio.event
async def disconnect():
    print('chatbot disconnected')


@sio.event(namespace='/chatbot')
async def get_chatbot_message(message):
    """

    :return:
    """
    print("Chatbot received message: " + message)
    gui_response_dict = GUI(message)
    gui_response = gui_response_dict['response']
    print('Chatbot sending message:', gui_response)
    await sio.emit('forward_chatbot', gui_response, namespace='/chatbot')

async def chatbot():
    await sio.connect(f'http://{HOST}:{PORT}', namespaces=['/', '/chatbot'])
    await sio.wait()


if __name__ == '__main__':
    asyncio.run(chatbot())
