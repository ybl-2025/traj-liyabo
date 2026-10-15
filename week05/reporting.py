"""Publication-style static figures and a computed Chinese experiment report."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from models import MODELS,DEFAULTS,acceleration,optimal_velocity
from experiments import scenario

COLORS={'OVM':'#d87828','FVDM':'#287fa0','IDM':'#7955a2'}
LABELS={'constant':'前车匀速','braking':'前车减速','emergency':'紧急制动压力测试'}


def configure():
    available={f.name for f in font_manager.fontManager.ttflist}
    fonts=[f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'] if f in available]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':fonts,'axes.unicode_minus':False,
        'figure.dpi':120,'savefig.dpi':170,'axes.spines.top':False,'axes.spines.right':False,
        'axes.facecolor':'#fafbfd','figure.facecolor':'white','grid.color':'#dbe1e8',
        'axes.grid':True,'grid.alpha':.65,'axes.axisbelow':True,'font.size':10,
        'axes.titlesize':12,'axes.titleweight':'bold','legend.frameon':False})


def save(fig,out,name):
    fig.savefig(Path(out)/name,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def table(frame,columns,labels=None):
    labels=labels or columns
    rows=['| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        vals=[]
        for key in columns:
            v=row[key]
            vals.append('—' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v))
        rows.append('| '+' | '.join(vals)+' |')
    return '\n'.join(rows)
