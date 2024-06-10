import pandas as pd
d=pd.DataFrame([['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
                ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
                ['Rainy','Cold','High','Strong','Warm','Change','No'],
                ['Sunny','Warm','High','Strong','Cool','Change','Yes']])
a=list(d.iloc[:,:-1].values)
b=list(d.iloc[:,-1].values)

s=['@' for i in range(len(a[0]))]
g=[['?' for i in range(len(a[0]))]for j in range(len(a[0]))] ##6*6 matrix
r=0
for i in range(len(b)):
    if b[i]=='Yes' and r==0:
        s=a[i].copy()
        r+=1
    if b[i]=='Yes' and r>0:
        for j in range(len(s)):
             if s[j]!=a[i][j]:
                 s[j]='?'
              
            
       
    if b[i]=='No':
        for k in range(len(g)):
            if s[k]!=a[i][k]:
                 g[k][k]=s[k]
            else:
                g[k][k]='?'

    for l in range(len(g)):
        if s[l]!=g[l][l]:
            g[l][l]=s[l]
        else:
            g[l][l]='?'

if g[i].count('?'):
        
        


                
   print(s)
   print(g)


                
        
