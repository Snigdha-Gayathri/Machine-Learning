import pandas as pd
d=pd.DataFrame([['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
                ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
                ['Rainy','Cold','High','Strong','Warm','Change','No'],
                ['Sunny','Warm','High','Strong','Cool','Change','Yes']])
a=list(d.iloc[:,:-1].values)
b=list(d.iloc[:,-1].values)
g=['@' for i in range(len(a[0]))]

for i in range(len(b)):
               if b[i]=='Yes' and i==0:
                     g=a[i]
               elif b[i]=='Yes' and i!=0:
                     for j in range(len(g)):
                         if g[j]!=a[i][j]:
                             g[j]='?'

                             
               if b[i]=='No' and i==0:
                     continue
               
               elif b[i]=='Yes' and i!=0:
                     g=a[i]
                     for j in range(len(g)):
                         if g[j]!=a[i][j]:
                             g[j]='?'


##               if b[i]=='Yes' and s==0:
##                         g=a[i].copy()
##                         s+=1
##               if b[i]=='Yes' and s>0:
##                     for j in range(len(g)):
##                         if g[j]!=a[i][j]:
##                             g[j]='?'
                   
print(g)
