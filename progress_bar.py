from dotenv import load_dotenv
import os
load_dotenv()

from bob_telegram_tools.bot import TelegramBot

bot = TelegramBot(os.getenv('comp_bot_token'),
                      os.getenv('chat_id'))
from tqdm.contrib.telegram import tqdm, trange
pb_iteration = 0
