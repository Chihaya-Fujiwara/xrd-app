import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import glob
import os

def motion(event):
    global gco, xdata, ydata, ind
    if gco is None:
        return
    x = event.xdata
    y = event.ydata 

    if x == None or y == None:
        return
    
    xdata[ind] = x
    ydata[ind] = y

    gco.set_data(xdata,ydata)    
    plt.draw()

def add(event):
    global gco, xdata, ydata, ind

    if event.button == 3:  # 左クリックで点を追加
        x_add = event.xdata
        y_add = event.ydata
        
        num = len(xdata[xdata < x_add])
        
        xdata = np.insert(xdata,int(num),x_add)
        ydata = np.insert(ydata,int(num),y_add)                
        
        gco.set_data(xdata,ydata)
        plt.draw()


def onpick(event):
    global gco, xdata, ydata, ind
    gco = event.artist
    xdata = gco.get_xdata()
    ydata = gco.get_ydata()
    ind = event.ind[0]

def release(event):
    global gco
    gco = None


gco = None     # ピックした要素が含まれる直線(Line2Dクラス)
ind = None     # ピックした直線のインデックス
xdata = None     # ドラッグするまえの直線のxデータを入れておく
ydata = None     

def filesearch(dir):
    path_list = glob.glob(dir + '/*')       # 指定dir内の全てのファイルを取得
  
    # パスリストからファイル名を抽出
    name_list = []
    for i in path_list:
        file = os.path.basename(i)          
        name, ext = os.path.splitext(file)  
        name_list.append(name)              
    return path_list, name_list



path_list, name_list = filesearch('data')

raw = pd.read_csv(path_list[0])

df = pd.read_csv('Bg_sbtracted_data/subtracted_data.csv')
name = 'name'
bins = 300

bgdata = []
for i in range(0,int(len(df)/bins),1):
    theta = df['2theta'][i*bins:i*bins+1].iloc[0]
    inte = df['intensity'][i*bins:i*bins+1].iloc[0]
    bgdata.append([theta,inte])

bg = pd.DataFrame(bgdata)
bg.columns = '0','1'

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

mag = (raw['intensity'][0:1]/bg['1'][0:1])[0]

x = bg['0']
y = bg['1']*mag

fig, ax = plt.subplots()
ax.plot(bg['0'],bg['1']*mag,"o-",picker=10,c='#FF0000')
#ax.scatter(raw['2theta'],raw['intensity'],s=1,picker=0,c='#0000FF')
ax.plot(raw['2theta'],raw['intensity'],lw=0.5,picker=0,c='#0000FF')
ax.set_xlabel('2 Theta')
ax.set_ylabel('Counts [a.u.]')

fig.canvas.mpl_connect('pick_event', onpick)
fig.canvas.mpl_connect('motion_notify_event', motion)
fig.canvas.mpl_connect('button_release_event', release)
fig.canvas.mpl_connect('button_press_event', add)

plt.show()

xdata = pd.DataFrame(xdata)
ydata = pd.DataFrame(ydata)

data = pd.concat([xdata,ydata],axis=1)
data.columns ='2theta','intensity'
data = data.sort_values(by='2theta')

fir = raw['2theta'][0:1].iloc[0]
step = raw['2theta'][1:2].iloc[0] - raw['2theta'][0:1].iloc[0]

df2 = raw.copy()
df2['intensity'] = df2['intensity']*0
for i in range(0,len(data),1):
    theta = data['2theta'][i]
    df2.loc[round((theta-fir)/step),'intensity'] =  data['intensity'][i]

df2 = df2.replace(0, np.nan)
df2['intensity'] = df2['intensity'].interpolate()
df2['intensity'] = df2['intensity'].interpolate(limit_area='outside', limit_direction='both')
raw['intensity'] = raw['intensity']-df2['intensity']

raw['intensity'][raw['intensity']<0]=0

raw.to_csv('Bg_sbtracted_data/bgsubtracted.csv',index=False)
df2.to_csv('Bg_sbtracted_data/subtracted_data.csv',index=False)
