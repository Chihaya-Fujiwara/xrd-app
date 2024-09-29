import streamlit as st
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import scipy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from plotly.subplots import make_subplots   
import subprocess
import plotly.express as px
import shutil
import glob
from PIL import Image
import time
import os
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


def filesearch(dir):
    path_list = glob.glob(dir + '/*')       # 指定dir内の全てのファイルを取得
  
    # パスリストからファイル名を抽出
    name_list = []
    for i in path_list:
        file = os.path.basename(i)          
        name, ext = os.path.splitext(file)  
        name_list.append(name)              
    return path_list, name_list


def bgprocess():
    
    gco = None     # ピックした要素が含まれる直線(Line2Dクラス)
    ind = None     # ピックした直線のインデックス
    xdata = None     # ドラッグするまえの直線のxデータを入れておく
    ydata = None     

    path_list, name_list = filesearch('data')

    raw = pd.read_csv(path_list[0])

    df = pd.read_csv('Bg_sbtracted_data/subtracted_data.csv"')
    name = 'name'
    bins = 100

    bgdata = []
    #for i in range(0,int(len(df)/bins),1):
    for i in range(0,int(len(df)/bins),1):
        #theta = df['2theta'][i*bins:(i+1)*bins].sum()/bins
        #inte = df['intensity'][i*bins:(i+1)*bins].sum()/bins
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

    df2 = raw.copy()
    df2['intensity'] = df2['intensity']*0
    for i in range(0,len(data),1):
        theta = data['2theta'][i]
        df2.loc[round((theta-10)/0.02),'intensity'] =  data['intensity'][i]

    df2 = df2.replace(0, np.nan)
    df2['intensity'] = df2['intensity'].interpolate()
    df2['intensity'] = df2['intensity'].interpolate(limit_area='outside', limit_direction='both')
    raw['intensity'] = raw['intensity']-df2['intensity']

    raw['intensity'][raw['intensity']<0]=0

    raw.to_csv('Bg_sbtracted_data/bgsubtracted.csv',index=False)
    df2.to_csv('Bg_sbtracted_data/subtracted_data.csv',index=False)



st.set_page_config(
    page_title="QAVIX",
    page_icon="atom_symbol", 
    layout="wide",
    #initial_sidebar_state="expanded",
)

@st.cache_data() 
def voigt(xval,params):
    norm,center,lw,gw = params
    z = (xval - center + 1j*lw)/(gw * np.sqrt(2.0))
    w = scipy.special.wofz(z)
    model_y = norm * (w.real)/(gw * np.sqrt(2.0*np.pi))
    return model_y

def simulate(struc):
    xrd_cond = XRDCalculator(wavelength='CuKa')
    xrd_calcd = xrd_cond.get_pattern(struc)
    dict_xrd_calcd = xrd_calcd.as_dict()
    calcd_x = xrd_calcd.x
    calcd_y = xrd_calcd.y
    calcd_hkls = xrd_calcd.hkls
    calcd_d = xrd_calcd.d_hkls
    x = np.arange(0,130,0.02)
    calcd_pattern = np.zeros(len(x))
    N = len(calcd_x)
    
    for i in range(N):
        norm = calcd_y[i]
        center = calcd_x[i]
        lw = 0.01
        gw = 0.01
        params = [norm, center, lw, gw]
        calcd_pattern += voigt(x, params)

    
    data = pd.DataFrame([x,calcd_pattern/calcd_pattern.max()]).T
    data_matome = pd.DataFrame([])
    
    #plt.close()
    return data, data, xrd_calcd

@st.cache_data() 
def plot_ref(cif_list):
    color_list = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3']
    color_list1 = px.colors.namespace.colorscale

    with st.expander("Candidate composition"):
        
        fig = make_subplots(rows=1, cols=1,
                shared_xaxes=True,
                vertical_spacing=0)
        
                
        fig.update_layout(legend=dict(x=0.99,
                            y=0.99,
                            xanchor='right',
                            yanchor='top',
                            ))
        
        for num in range(0,len(cif_list)-1,1):
            struc_name = cif_list[num].split(':')[0]
            chemical_formula = cif_list[num].split(':')[1]
            #st.write(cif_list[num].split(':'))
            path = str(cif_path)+'/cif_data/'+str(struc_name)+'.cif'
            parser = CifParser(path)
            struc = parser.get_structures(primitive=False, symmetrized=False)[0]
            # simulation 実行
            data,fig1,calc_data = simulate(struc)
            data_all = pd.DataFrame([calc_data.x,calc_data.y,calc_data.hkls,calc_data.d_hkls]).T

            detail = []
            for j in range(0,len(data_all),1):
                x_ = data_all[0][j]
                y_ = data_all[1][j]
                for k in range(0,len(data_all[2][j]),1):
                    hkl = data_all[2][j][k]['hkl']
                    m = data_all[2][j][k]['multiplicity']
                    detail.append([x_,y_,hkl,m])
                    
            detail = pd.DataFrame(detail)
            detail.columns = '2theta','intensity', 'hkl','m'
            data.columns = '2theta','intensity'
            
            fig.add_trace(go.Scatter(
            x= data['2theta'],
            y= data['intensity'],
            marker_color=color_list[num],
            name = str(struc_name)+'.cif :' + str(chemical_formula)
            ), row=1, col=1)

        fig.update_xaxes(title=dict(text='2 Theta [theta]'),mirror=True, row=1, col=1)
        selected_points = plotly_events(fig)
    
    return data, fig    


############ all plot #########################
@st.cache_data() 
def plot_all_with_exp(cif_list,expdataall,times):

    if int(len(cif_list))==0:
        
        color_list = px.colors.qualitative.Plotly
        color_list1 = px.colors.sequential.Turbo
                
        fig_ = make_subplots(rows=1, cols=1,
                shared_xaxes=True,
                vertical_spacing=0)
        
        fig_.update_layout(legend=dict(x=0.99,
                            y=0.99,
                            xanchor='right',
                            yanchor='top',
                            ))

        
        c = 0
        for num_num_num in range(0,len(expdataall),1):
            for num_num in range(0,len(expdataall[num_num_num].columns)-1,1):
                fig_.add_trace(go.Scatter(
                        x= expdataall[num_num_num]['2theta'],
                        y= expdataall[num_num_num].iloc[:,num_num+1]+times*c,
                        name = expdataall[num_num_num].columns[num_num+1],
                        marker_color=color_list1[c]
                        ), row=1, col=1,
                        )
                c = c+1

        fig_.update_xaxes(title=dict(text='2 Theta [theta]'),mirror=True, row=1, col=1)
        fig_.update_yaxes(title=dict(text='Intensity [a.u.]'),mirror=True, row=1, col=1)
        fig_.update(layout_xaxis_range = [0,120])
        fig_.update_layout(width=800,height=400)
        
        data_all_all = []
        detail_all = []
        
        selected_points = plotly_events(fig_)
            
        return data_all_all, fig_, detail_all    
        
                
    else:
        color_list = px.colors.qualitative.Plotly
        color_list1 = px.colors.sequential.Turbo
            
        fig = make_subplots(rows=2, cols=1,
                row_heights=[0.8,0.2],
                shared_xaxes=True,
                vertical_spacing=0)
        
        fig.update_layout(legend=dict(x=0.99,
                            y=0.99,
                            xanchor='right',
                            yanchor='top',
                            ))


        c = 0
        for num_num_num in range(0,len(expdataall),1):
            for num_num in range(0,len(expdataall[num_num_num].columns)-1,1):
                fig.add_trace(go.Scatter(
                        x= expdataall[num_num_num]['2theta'],
                        y= expdataall[num_num_num].iloc[:,num_num+1]+times*c,
                        name = expdataall[num_num_num].columns[num_num+1],
                        marker_color=color_list1[c]
                        ), row=1, col=1,
                        )
                c = c+1
        
        data_all_all = []
        detail_all = []
        
    
        for num in range(0,len(cif_list),1):
            Dir_cif = cif_list[num].split(':')[1] 
            struc_name = cif_list[num].split(':')[0]
            chemical_formula = cif_list[num].split(':')[2]
            #st.write(cif_list[num].split(':'))
            path = str(cif_path)+'/cif_data/'+str(Dir_cif)+'/'+str(struc_name)+'.cif'
            parser = CifParser(path)
            struc = parser.get_structures(primitive=False, symmetrized=False)[0]

            data,fig1,calc_data = simulate(struc)

            data_all = pd.DataFrame([calc_data.x,calc_data.y,calc_data.hkls,calc_data.d_hkls]).T

            detail = []
            for j in range(0,len(data_all)-1,1):
                x_ = data_all[0][j]
                y_ = data_all[1][j]
                for k in range(0,len(data_all[2][j]),1):
                    hkl = data_all[2][j][k]['hkl']
                    m = data_all[2][j][k]['multiplicity']
                    detail.append([x_,y_,hkl,m])
                    
            detail = pd.DataFrame(detail)
            detail.columns = '2theta','intensity', 'hkl','m'
            data.columns = '2theta','intensity'  
            data_all_all.append([str(struc_name),str(chemical_formula),data])
            detail_all.append([str(struc_name),str(chemical_formula),detail])
            
            fig.add_trace(go.Scatter(
            x= data['2theta'],
            y= data['intensity'],
            marker_color=color_list[num],
            name = str(struc_name)+'.cif :' + str(chemical_formula)
            ), row=2, col=1)
        
        fig.update_xaxes(title=dict(text='2 Theta [theta]'),mirror=True, row=2, col=1)
        fig.update_yaxes(title=dict(text='Intensity [a.u.]'),mirror=True, row=1, col=1)
        fig.update(layout_xaxis_range = [0,120])
        fig.update_layout(width=800,height=500)
            
        selected_points = plotly_events(fig)
        
        return data_all_all, fig, detail_all    

        #except ValueError:
        #    st.warning(str(struc_name)+'.cif :' + str(chemical_formula) + 'cannot be ploted.', icon="⚠️")

#@st.cache_data(experimental_allow_widgets=True)    
def figopen(fig):
    fig.update_layout(width=1200,height=600)
    #selected_points = plotly_events(fig)
    fig.show()

@st.cache_data() 
def cifopen(cif_list,vestapath):
    vestapath = "C:\\Users\\chiha\\OneDrive\\11_download\\VESTA-win64\\VESTA-win64\\VESTA.exe"
    subprocess.Popen([vestapath])
    subprocess.Popen([vestapath, '-open', str(cif_path)+'/cif_data/'+str(cif_list)+'.cif'])    

@st.cache_data()
def search(options,type,df_):
    for i in range(0,len(options),1):
        df_ = df_[df_['chemical species'].str.contains('"'+options[i]+'"')]
    
    if type == "Only":
        df_ = df_[df_['Number of chemical species']==len(options)]
    else:
        df_ = df_
    
    data_df = pd.DataFrame(
    {"favorite": [False for i in range(0,len(df_),1)],
    }
    )
    
    df_ = df_.reset_index(drop=True)
    data_cif = pd.concat([data_df,df_],axis=1)
    
    with st.expander("Candidate composition"):
        candidate = st.data_editor(
        data_cif,
        column_config={
        "favorite": st.column_config.CheckboxColumn(
            "checkbox",
            help="Select your **favorite** widgets",
            default=False,
        )
        },
        disabled=['COD Number','Chemical formula', 'Chemical moiety','chemical formula structural','chemical species','Number of chemical species','DOI'],
        hide_index=True,
        )   
          
        candidate = candidate[candidate['favorite']==True]
        candidate = candidate.reset_index(drop=True)
    
    return candidate

@st.cache_data
def serach_2(options,type,df_):
    for i in range(0,len(options),1):
        df_ = df_[df_['chemical species'].str.contains('"'+options[i]+'"')]

    if type == "Only":
        df_ = df_[df_['Number of chemical species']==len(options)]
    else:
        df_ = df_
    
    df_df = df_.loc[:, ['COD Number','Dir','Chemical formula']]

    list_df_ = df_df.values.tolist()

    return list_df_

    
@st.cache_data
def convert_df(df): 
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data
def rasdata(uploaded_file):
    df = pd.read_csv(uploaded_file,encoding='shift_jis')
    st.write(uploaded_file.name)
    start_index_data = df[df['*RAS_DATA_START'] == '*RAS_INT_START'].index
    fin_index_data =df[df['*RAS_DATA_START'] == '*RAS_INT_END'].index
    r_d = ['r_d'+str(i) for i in range(0,len(start_index_data),1)]
    s_d = ['s_d'+str(i) for i in range(0,len(start_index_data),1)]
    
    for i in range(0,len(start_index_data),1):
        r_d[i] =  df[start_index_data[i]+1:fin_index_data[i]]
        s_d[i] = []
        for j in range(0,len(r_d[i]),1):
            r_d[i]['*RAS_DATA_START'][0:1].iloc[0].split(' ')
            s_d[i].append([float(r_d[i]['*RAS_DATA_START'][j:j+1].iloc[0].split(' ')[0]),float(r_d[i]['*RAS_DATA_START'][j:j+1].iloc[0].split(' ')[1])])
        s_d[i] = pd.DataFrame(s_d[i])
        s_d[i] = s_d[i].reset_index(drop=True)

        s_d[i].columns = '2theta',str(uploaded_file.name)+'_exp_'+str(i)

        if i == 0:
            data = s_d[i]
        else:
            data = pd.concat([data,s_d[i][str(uploaded_file.name)+'_exp_'+str(i)]],axis=1)
            
    return data

def csvdata(uploaded_file):
    st.write(str(uploaded_file.name))
    df = pd.read_csv(uploaded_file,encoding='shift_jis')
    df_ = pd.concat([df['2theta'],df['intensity']],axis=1)
    df_.columns = '2theta', str(uploaded_file.name)+'_exp_0'
    return df_

def rasdata_temp(temp_data):
    df = pd.read_csv(temp_data,encoding='shift_jis')
    st.write('Example data is loaded.')
    start_index_data = df[df['*RAS_DATA_START'] == '*RAS_INT_START'].index
    fin_index_data =df[df['*RAS_DATA_START'] == '*RAS_INT_END'].index
    r_d = ['r_d'+str(i) for i in range(0,len(start_index_data),1)]
    s_d = ['s_d'+str(i) for i in range(0,len(start_index_data),1)]
    
    for i in range(0,len(start_index_data),1):
        r_d[i] =  df[start_index_data[i]+1:fin_index_data[i]]
        s_d[i] = []
        for j in range(0,len(r_d[i]),1):
            r_d[i]['*RAS_DATA_START'][0:1].iloc[0].split(' ')
            s_d[i].append([float(r_d[i]['*RAS_DATA_START'][j:j+1].iloc[0].split(' ')[0]),float(r_d[i]['*RAS_DATA_START'][j:j+1].iloc[0].split(' ')[1])])
        s_d[i] = pd.DataFrame(s_d[i])
        s_d[i] = s_d[i].reset_index(drop=True)

        s_d[i].columns = '2theta','template_exp_'+str(i)

        if i == 0:
            data = s_d[i]
        else:
            data = pd.concat([data,s_d[i]['template_exp_'+str(i)]],axis=1)
    
    return data



@st.cache_data()
def fileread(uploaded_file):
    data_exp = ['dataexp'+str(gi) for gi in range(0,len(uploaded_file),1)]
    expdataall = []
    
    for gj in range(0,len(uploaded_file),1):
        filetype = uploaded_file[gj].name.split('.')[-1]
        
        if filetype == 'ras':
            data_exp[gj] = rasdata(uploaded_file[gj])
            expdataall.append(data_exp[gj])
        
        elif filetype=='csv':
            data_exp[gj] = csvdata(uploaded_file[gj])
            expdataall.append(data_exp[gj])
        
        else:
            st.write('error')
            
    return expdataall




def main(database): 
    ccc1,ccc2,ccc3 = st.columns([1,3,1])
    ccc2.image(image1) 
    ccc2.subheader("")
    st.write('This is an application for quick analysis and visualization of powder XRD data. Read the following documents before using this software.  This software is created by Chihaya Fujiwara (https://researchmap.jp/C_Fujiwara or -----). If you have any questions, please contact this email address ---[at]--.')
    with st.expander('Quick Summary'):
        markdown2 = """        
        * **Exp file** ： You can load a .ras file, which is the experimental X-ray diffraction data from RIGAKU instrument. You can also convert .ras files to .csv files here.
        * **Candidate composition** ： Please select the elements that are considered to be included. These database displayed here is based on **Crystallography Open Database (COD)** (https://www.crystallography.net/cod/). If you would like to get more information of the database, please access the url(https://www.crystallography.net/cod/search.html) and enter the corresponding COD ID. 
            (For example 9006388 : NaCl).
        * **Candidate List** ：Here you can select the data for which you want the XRD diffraction pattern to be displayed. 
        * **XRD pattern** :  The XRD pattern of the actual measurement loaded can be compared with the XRD pattern of the selected crystal structure. These Simulations are based on the **pymategen** python library (https://pymatgen.org/pymatgen.analysis.diffraction.html). 
        * **Exp data** : You can save the experimental XRD resutls as .csv files.
        * **Cif data** : You can check the reference cif file by interfacing it with **VESTA** (https://jp-minerals.org/vesta/jp/). The XRD simulation pattern can also be saved as a csv file.
        * **Background** : You can perform a various background processes here.
        * **Upload cif file** : CIF files can be optionally added to the database. Just upload the ciffile and press the button database upload..
        """

        st.markdown(markdown2)        

             
    with st.sidebar.expander("Select"):
            g = st.selectbox("Tool type", ('Search','Add cif','Linking with VESTA'))
            
    if g == 'Linking with VESTA':
        new_vesta_path = st.text_input('Input VESTA PATH')
        updata_vesta_path = read(path=v_path)[0]
        st.write('CURRENT VESTA PATH:  '+updata_vesta_path)
        updata_vesta_path = new_vesta_path
        st.write('NEW VESTA PATH:  '+updata_vesta_path)
        if st.button('UPDATE VESTA PATH'):
            with open('vestapath.txt', 'w') as f:
                for d in updata_vesta_path:
                    f.write(d)
    
    if g == 'Search':
        with st.sidebar.expander('Exp file'):    
                uploaded_file = st.file_uploader("Choose a .ras file",accept_multiple_files=True)
                if not uploaded_file:
                    st.write('Example file')
                    expdataall = [rasdata_temp('example/example.ras')]
                    
                else:
                    expdataall = fileread(uploaded_file)
                    

        
        with st.expander("Candidate composition"):
            
            col1,col2,col3 = st.columns([3,3,1])
        
            options = col1.multiselect(
            "Select elements",
            ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
            'Zr','Nb','Mo','Tc','Ru','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd',
            'Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac',
            'Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
            ) 
            
            type = col2.select_slider(
                "Select serach type",
                options=[
                    "At least",
                    "Only",
                ],
            )

            list_df_ = serach_2(options,type,database)
            col1,col2 = st.columns([6,1])  
            candi = col1.multiselect(
            "Choose the composition",
            list_df_)
        
            if 'count' not in st.session_state: 
                st.session_state.count = ''

            if col2.button('Move to List'):
                for j in range(0,len(candi),1):
                    if str(candi[j][0]) not in st.session_state["count"]:
                        st.session_state.count += str(candi[j][0])+':'+str(candi[j][1])+':'+str(candi[j][2])+',' #値の更新
        
        
        with st.expander("Candidate List"):

            col1,col2 = st.columns([6,1])  
            cif_list = st.session_state.count.split(',')[:-1]
            
            if 'ciflist' not in st.session_state:
                st.session_state.ciflist = ''
                pre_cif = []
                
            if len(st.session_state.ciflist) == 0:
                cif_list_list = col1.multiselect(
                "Choose the crystal data",
                    cif_list)                
            else:
                pre_cif = st.session_state.ciflist.split(',')[:-1]
                cif_list_list = col1.multiselect(
                "Choose the crystal data",
                    cif_list,pre_cif)

            for j in range(0,len(cif_list_list),1):
                if str(cif_list_list[j]) not in st.session_state.ciflist:
                    st.session_state.ciflist += cif_list_list[j]+','
            
                
            if col2.button('All clear (push twice)'):
                st.session_state.count = ''  
                st.session_state.ciflist = ''


        try:
            with st.expander("XRD pattern"):

                times = st.slider('Shift the baseline',000,10000,1000)
                data, fig, detail = plot_all_with_exp(cif_list_list,expdataall,times)       
                
                #st.write(detail)  
                col1,col2,col3= st.columns([1,1,1])       
                
                if col1.button('Expand Selected Figure'):
                    figopen(fig)


        except IndexError:
            st.write('choose reference')
        except FileNotFoundError:
            st.write('choose reference')
        except TypeError:
            pass
        
        col11,col12 = st.columns(2)

        with col11.expander("Exp data"):
            namelist = []
            for p in range(0,len(expdataall),1):
                for k in range(0,len(expdataall[p].columns[1:]),1):
                    namelist.append([expdataall[p].columns[1+k:2+k][0],p,k])
            
            namelist = pd.DataFrame(namelist)
            
            expdata_ = st.selectbox(
            "Select exp data",
            namelist)
            
            name = namelist[namelist[0]==expdata_]
            x = expdataall[name[1].iloc[0]].iloc[:,0]
            y = expdataall[name[1].iloc[0]].iloc[:,name[2].iloc[0]+1]
            expdata_1 = pd.concat([x,y],axis=1)
            expdata_1.columns = '2theta','intensity'
            
            expdata_1_up = convert_df(expdata_1)
            
            st.download_button(
                    label="Save data",
                    data= expdata_1_up,
                    file_name=expdata_+'_raw.csv',
                    mime="csv",
                    )

                    
        with st.expander("Background"):
            namelist_ = []
            cc1,cc2 = st.columns([5,1])
            for p in range(0,len(expdataall),1):
                for k in range(0,len(expdataall[p].columns[1:]),1):
                    namelist_.append([expdataall[p].columns[1+k:2+k][0],p,k])
            
            namelist_ = pd.DataFrame(namelist_)
            
            expdata_ = cc1.selectbox(
            "Select experimental data",
            namelist_)
            
            name = namelist[namelist[0]==expdata_]
            x = expdataall[name[1].iloc[0]].iloc[:,0]
            y = expdataall[name[1].iloc[0]].iloc[:,name[2].iloc[0]+1]
            expdata_1 = pd.concat([x,y],axis=1)
            expdata_1.columns = '2theta','intensity'
            
            if cc2.button('Background update'):
                expdata_1.to_csv('data/data.csv',index=False)
                result = subprocess.run(['Bg.bat'], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)                
                
            #exp_bgsbtructed = pd.read_csv('Bg_sbtracted_data/bgsubtracted.csv')
            bg = pd.read_csv('Bg_sbtracted_data/subtracted_data.csv')
            
            exp_bgsbtructed = pd.concat([expdata_1['2theta'],expdata_1['intensity']-bg['intensity']],axis=1)
            exp_bgsbtructed['intensity'][exp_bgsbtructed['intensity']<0]=0
            
            fig2 = make_subplots(rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0)
    
    
            fig2.update_layout(legend=dict(x=0.99,
                        y=0.99,
                        xanchor='right',
                        yanchor='top',
                        ))
            
            fig2.add_trace(go.Scatter(
            x= expdata_1['2theta'],
            y= expdata_1['intensity'],
            name = expdata_,
            marker_color = '#0000FF',
            ), row=1, col=1)
            
            fig2.add_trace(go.Scatter(
            x= bg['2theta'],
            y= bg['intensity'],
            name = "Background",
            marker_color = 	'#FF0000'
            ), row=1, col=1)
            
            fig2.add_trace(go.Scatter(
            x= exp_bgsbtructed['2theta'],
            y= exp_bgsbtructed['intensity'],
            name = expdata_+'_bgsubtructed.csv',
            marker_color = 	'#000080'
            ), row=2, col=1)
    
            fig2.update_xaxes(title=dict(text='2 Theta [theta]'),mirror=True, row=2, col=1)
            fig2.update_yaxes(title=dict(text='Intensity [a.u.]'),mirror=True, row=1, col=1)
            fig2.update(layout_xaxis_range = [0,120])
            fig2.update_layout(width=800,height=450)

            #fig2.update(layout_yaxis_range = [0,expdata_1['intensity'].max()*2])

            selected_points = plotly_events(fig2)
            
            expdata_1_update = convert_df(expdata_1)
            bg_update = convert_df(bg)
            exp_bgsbtructed_update = convert_df(exp_bgsbtructed)
            
            c1,c2,c3,c4 = st.columns([2,3,2,3])
            
            if c1.button('Expand fig'):
                figopen(fig2)
            
            c2.download_button(
                    label="Save exp raw data",
                    data= expdata_1_update,
                    file_name=expdata_+'_raw.csv',
                    mime="csv",
                    )
            
            c3.download_button(
                    label="Save Bg data",
                    data= bg_update,
                    file_name=expdata_+'_Bg.csv',
                    mime="csv",
                    )  
            
            c4.download_button(
                    label="Save Bgsbtructed data",
                    data= exp_bgsbtructed_update,
                    file_name=expdata_+'_Bgsbtructed.csv',
                    mime="csv",
                    )                   
                 


        
        with col12.expander("Cif open"):
            cif = st.selectbox(
            "Select cif file",
            cif_list_list)
            
            try:
                data = pd.DataFrame(data)
                data_ = data[data[0]==cif.split(':')[0]]
                data_ex = data_[2].iloc[0]
                data_ex = convert_df(data_ex)
                detail = pd.DataFrame(detail)
                detail_ = detail[detail[0]==cif.split(':')[0]]
                detail_ex = detail_[2].iloc[0]
                detail_ex = convert_df(detail_ex)
            
            except KeyError:
                st.write('select data')
                
            except UnboundLocalError:
                st.error('Excessively many experimental data are loaded.', icon="🚨")
                st.stop()


            try:
                st.write('Chemical formula = '+ cif.split(':')[2])
                st.write('COD ID = '+cif.split(':')[0])
                
                col1,col2,col3 = st.columns(3)
                if col1.button('See Crystal Structure By VESTA'):
                        cifopen(cif.split(':')[1]+'/'+cif.split(':')[0],vestapath)
                        
                col2.download_button(
                label="Save reference data",
                data= data_ex,
                file_name=cif.split(':')[2]+'.csv',
                mime="csv",
                )
                
                col3.download_button(
                label="Save reference hkl data",
                data= detail_ex,
                file_name=cif.split(':')[2]+'_hkl.csv',
                mime="csv",
                )
                subprocess.run("clip", input=str(cif.split(':')[0]), text=True)
                
            except AttributeError:
                col1.write('choose cif file')
        
            st.write('COD: https://www.crystallography.net/cod/search.html)')
               
    if g == 'Add cif':
        uploaded_file = st.file_uploader("Upload cif file",accept_multiple_files=True)
        if st.button('Database updata'):
            for i in range(0,len(uploaded_file),1):
                bytes_data = uploaded_file[i].getvalue().decode("utf-8")
                with open(cif_path+'/cif_add/'+uploaded_file[i].name+'.cif', 'w') as f:
                    for d in bytes_data:
                        f.write(d)
            
            cifadd(uploaded_file)

def cifadd(uploaded_file):
    cif_data = []
    cif_data = cif_data+glob.glob(cif_path+"/cif_add/*.cif", recursive=True)
    cif_data_exist = cif_data+glob.glob(cif_path+"/cif_data/52/*.cif", recursive=True)

    def read(path):
        with open(path) as f:
            lines = f.readlines()
        lines_strip = [line.strip() for line in lines]
        return lines_strip

    database = pd.read_csv(str(cif_path)+'/database.csv')
    num_start = database['0'][-1:].iloc[0]
    
    df = []
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    percent_complete = 0
    total_cnt = len(uploaded_file)
    
    for j in range(0,len(cif_data),1):        
        
        try:
            s = str(cif_data[j].split("\\")[-1])
            re = database[database['4']==s]['0'].iloc[0]
            
        except IndexError:
                
            parser = CifParser(cif_data[j])
            structure = parser.get_structures()[0]

            num = int(num_start) + j + 1
        
            data = read(cif_data[j])
            
            doi =  [s for s in data if '_journal_paper_doi' in s]
            
            if doi == []:
                doi = 'check cod number'
            else:
                doi = doi[0].split(' ')[-1]
            
            chem_name = structure.formula
            
            for k in range(0,10,1):
                chem_name = chem_name.replace(str(k),'')

            comp = chem_name.split(' ')
            
            comp_comp = ''
            for q in range(0,len(comp),1):
                comp_comp = comp_comp +'"'+comp[q]+'"'+','
            
            st.write(cif_data[j].split("\\")[-1])
            doi = cif_data[j].split("\\")[-1]          
                
            df.append([num,structure.formula,comp_comp,len(comp),doi,52])
            new_path = shutil.copy(cif_data[j],cif_path+'/cif_data/52/'+str(num)+'.cif')
            
            percent_complete = percent_complete + 1/total_cnt
            if percent_complete <= 1:
                my_bar.progress(percent_complete, text=progress_text)
                
    try:          
        df = pd.DataFrame(df)
        df.columns = '0','1','2','3','4','5'
        data_updata = pd.concat([database,df],axis=0)
        data_updata.to_csv(cif_path+'/database.csv',index=False)
        st.write('Updata cif data')
        
    except ValueError:
        st.write('Updata no data')
        

@st.cache_data()
def vesta_read(path):
    with open(path) as f:
        lines = f.readlines()
        lines_strip = [line.strip() for line in lines]
    return lines_strip


def read(path):
    with open(path) as f:
        lines = f.readlines()
    lines_strip = [line.strip() for line in lines]
    return lines_strip           

@st.cache_data()
def readdatabase(cif_path):
    database = pd.read_csv(str(cif_path)+'/database.csv')
    database.columns = 'COD Number','Chemical formula','chemical species','Number of chemical species','DOI','Dir'
    st.write(str(len(database))+' files are in this database now.')
    return database

@st.cache_data()
def readfigure(cif_path):
    image1 = Image.open('picture.png')
    return image1

if __name__ == '__main__':
    cif_path =  os.getcwd()
    v_path = 'vestapath.txt'
    vestapath = vesta_read(v_path)[0]
    st.write('VESTA PATH : ' + str(vestapath))
    database = readdatabase(cif_path)
    image1 = readfigure(cif_path)
    main(database)

