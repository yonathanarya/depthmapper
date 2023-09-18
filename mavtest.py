from lib.mavlink import sendDepth
import configparser

config = configparser.ConfigParser()
config.read('settings.conf')
sendDepth(config)