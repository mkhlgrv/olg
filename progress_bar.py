from dotenv import load_dotenv
import os
load_dotenv()
import telebot
from telebot.types import InputMediaPhoto
import io 
comp_bot_token = os.environ.get('comp_bot_token')
chat_id = os.environ.get('chat_id')
bot = telebot.TeleBot(comp_bot_token)

def send_figure(plot, message=None):
    try:
        
        buf = io.BytesIO()
        plot.fig.savefig(buf, format="jpg")

        if message is None:
            message = bot.send_photo(chat_id,buf.getvalue())
        else:
            bot.edit_message_media(message_id=message.id, chat_id=chat_id, media = InputMediaPhoto(buf.getvalue()))
    except:
        pass
    return message

from tqdm.contrib.telegram import tqdm, trange
pb_iteration = 0
