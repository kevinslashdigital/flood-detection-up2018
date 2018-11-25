import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Model support
def train_model():
      
  df = pd.read_csv("dataset/flood_data.csv", sep = ",",usecols=['created_at','sensor_location','streamHeight'])
  # df['updated_at'] = pd.to_datetime(df['updated_at'],infer_datetime_format=True)

  df['created_at'] = pd.to_datetime(df['created_at'],utc=False)
  df.sort_values(by='created_at',ascending=True,inplace=True,na_position='first') # This
  # df = df.cumsum()
  print(df.head(20))

  df.set_index('created_at')

  df_sr = df.query('sensor_location == "Siemreap"')
  df_bbt = df.query('sensor_location == "Battambang"')
  df_kc = df.query('sensor_location == "Kampong Chhnang"')
  df_ps = df.query('sensor_location == "Pursat"')


  print(df_sr.head(20))

  df_sr.plot(x='created_at',y='streamHeight',title='Siemreap').figure.savefig('output/sr.png')
  df_bbt.plot(x='created_at',y='streamHeight',title='Battambang').figure.savefig('output/btb.png')
  df_kc.plot(x='created_at',y='streamHeight',title='Kampong Chhnang').figure.savefig('output/kc.png')
  df_ps.plot(x='created_at',y='streamHeight',title='Pursat').figure.savefig('output/ps.png')
  plt.show()
    
if __name__ == "__main__":
      
    # df = pd.DataFrame( 
    #   {
    #     'Symbol':['A','A','A'] ,
    #     'Date':['02/20/2015','01/15/2016','08/21/2015']
    #   })
    # print(df)

    # df['Date'] =pd.to_datetime(df.Date)
    # df.sort_values(by='Date',inplace=True, ascending=True)
    # print(df)

    # construct the argument parse and parse the arguments
    train_model()
