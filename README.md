# AI-Bitcoin-Trading-Bots

### Description
These are the files with code related to the AI project. Here you can find all the implementations and tests of different strategies.   

### Notes  
Get historical data.ipynb is just a demonstration how the data was gathered using the Binance API, you won't be able to run this file without an API key. The data itself is uploaded to this repository under the title "bitcoin.csv".

Also all trading bots are connected with one another with a specific logic. For the best understanding you should check the files in the specified order:  
1. Price&VolumeLongBot.ipynb  
2. Price&VolumeLongShortBot.ipynb  
3. TripleSMACrossoverLongBot.ipynb  
4. TripleSMACrossoverLongShortBot.ipynb  
5. BOLL_RSI_MACD_LongBot.ipynb  
6. BOLL_RSI_MACD_LongShortBot.ipynb  
7. RNN_Long_bot.py  
8. RNN_LongShort_bot.py

### RNN bots run instructions
In order to run RNN_Long_bot or RNN_LongShort_bot you should just run the responsible files.
The result displayed in the terminal will be multiple coefficient using the bot and using Buy&Hold strategy.

There is also file named RNN_model. It was used to create the model, the bots are working with. You can create and save your own model by changing parameters in the file. If you want to use bots your model, change pathfile from default value to the name of your model.
