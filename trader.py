from alpaca.trading.client import TradingClient

api_key = 'CKTUR8QA4H4XNX9YM8FS'
secret_key = 'Kgd79zUduwgzNBXgfSFaSQTZi87y4M58CXbOX3Is'

trading_client = TradingClient('api-key', 'secret-key', paper=True)

account = trading_client.get_account()