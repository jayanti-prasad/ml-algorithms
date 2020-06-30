import sys
import pandas as pd

def get_frame (X0, look_back):
   dF = pd.DataFrame()
   for i in range(0, look_back):
      col = "X_"+str(i) 
      XX =  X0.iloc[i:] 
      dF[col] = XX 
      dF[col] = dF[col].shift(-i)
    
   return dF.iloc[:-look_back]   

if __name__ == "__main__":

   df = pd.read_csv(sys.argv[1])
   X0 = df['Passengers']
   look_back = 4
   dF = get_frame (X0, look_back)

   print(dF.shape) 
   dF.to_csv("frame.csv")
