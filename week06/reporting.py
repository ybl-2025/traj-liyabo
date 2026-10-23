"""Exportable scientific figures for the six mixed circle scenarios."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from models import MODELS,LABELS,headings

COLORS=dict(GOAL='#9a6b6b',DIST='#d58a22',PRED='#4779c4',SIDE='#178b79')
STATUS=dict(success='完成且无重叠/越界',collision='发生重叠',boundary='发生越界',timeout='时限内未完成')
PED_COLORS=['#178b79','#2f9fb0','#557bc1','#9274b3','#1c6f91','#79a641']
CAR_COLORS=['#d35a56','#bd4c65','#dd7c42','#ae4138','#b9628c','#c88835']
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'#f5f7fb','axes.facecolor':'white','savefig.facecolor':'#f5f7fb'})

def save(fig,out,name,note=''):
    fig.get_layout_engine().set(rect=(0,.05,1,.94),h_pad=.12,w_pad=.07,hspace=.10,wspace=.05)
    if note:fig.text(.02,.012,note,fontsize=8,color='#626f80')
    fig.savefig(out/name,dpi=150,bbox_inches='tight');plt.close(fig)

def snapshot(ax,r,a,scene,time=3.5,width=3.5,full=False,labels=True,initial=False):
    k=0 if initial else int(np.argmin(abs(r['time']-time)));x=r['position'][k]
    cars=np.flatnonzero(a.half_box[:,0]>0);peds=np.flatnonzero(a.half_box[:,0]==0);heading=headings(a)
    if scene=='narrow':
        ax.axhspan(-width/2,width/2,color='#e8edf3')
        for s in [-1,1]:ax.axhline(s*width/2,color='#526174',lw=1.1)
        ax.axhline(0,color='#bac3ce',ls='--',lw=.7);ax.set(xlim=(9,25),ylim=(-width/2-.5,width/2+.7))
    else:
        ax.add_patch(Circle((0,0),10,fill=False,ls='--',ec='#b8c2ce',lw=.8))
        ax.set(xlim=(-12.8,12.8),ylim=(-12.8,12.8))
    for j in range(len(x)):
        is_car=j in cars;rank=int(np.flatnonzero((cars if is_car else peds)==j)[0]);color=(CAR_COLORS if is_car else PED_COLORS)[rank%6]
        if full:ax.plot(r['position'][:,j,0],r['position'][:,j,1],c=color,alpha=.35,lw=.8)
        if not initial:ax.plot(r['position'][:k+1,j,0],r['position'][:k+1,j,1],c=color,lw=1.2)
        if is_car:
            h=heading[j];left=np.array([-h[1],h[0]])
            corners=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*a.half_box[j]
            corners=x[j]+corners[:,0,None]*h+corners[:,1,None]*left
            ax.add_patch(Polygon(corners,fc=color,ec='#7c3434',alpha=.78,zorder=5))
            ax.arrow(*(x[j]+h*.8),*(h*.8),head_width=.22,head_length=.32,color='#fff4ed',length_includes_head=True,zorder=6)
            if labels:ax.text(*x[j],f'C{rank+1}',ha='center',va='center',fontsize=7,color='white',zorder=7)
        else:
            ax.add_patch(Circle(x[j],.25,fc=color,ec='#183b47',zorder=6))
            if labels:ax.annotate(f'P{rank+1}',x[j]+[0,-.6],ha='center',fontsize=7,zorder=7)
    ax.set_aspect('equal');ax.set(xlabel='x（m）',ylabel='y（m）')

def make_figures(out,runs,agents,metrics,steps):
    fig,axes=plt.subplots(2,3,figsize=(16,11),layout='constrained')
    for n,ax in enumerate(axes.flat,1):
        key=f'circle_n{n}_SIDE';m=metrics[key]
        snapshot(ax,runs[key],agents[key],'circle',time=3.5,full=True)
        ax.set_title(f"{n} 人＋{n} 车｜SIDE\n{STATUS[m['status']]} · 到达 {m['arrival_fraction']:.0%}",fontsize=11)
    fig.suptitle('场景二六组总览｜1 人 1 车 → 6 人 6 车',fontsize=23,fontweight='bold')
    save(fig,out,'summary.png','固定半径 10 m、固定参数；浅色为完整诊断轨迹，实体快照 t=3.5 s。车辆方向按各自对径路线绘制；重叠后结果不是安全通行。')

    fig,axes=plt.subplots(2,3,figsize=(16,11),layout='constrained')
    for n,ax in enumerate(axes.flat,1):
        key=f'circle_n{n}_SIDE';a=agents[key]
        snapshot(ax,runs[key],a,'circle',initial=True)
        for j in range(n):
            ax.plot([a.position[j,0],a.goal[j,0]],[a.position[j,1],a.goal[j,1]],c=PED_COLORS[j],ls=':',lw=.7,alpha=.6)
        ax.set_title(f'{n} 人＋{n} 车｜共 {2*n} 个主体',fontweight='bold')
    fig.suptitle('六组初始布局｜同编号 P 与 C 从圆周对径出发',fontsize=21,fontweight='bold')
    save(fig,out,'circle_layouts.png','所有 2N 个起点沿圆周均匀分布；车在右半圆，配对行人在左半圆，均以对径点为目标。虚线圆不构成实体墙。')

    fig,axes=plt.subplots(6,4,figsize=(18,26),layout='constrained')
    for n in range(1,7):
        for j,m in enumerate(MODELS):
            key=f'circle_n{n}_{m}';row=metrics[key]
            snapshot(axes[n-1,j],runs[key],agents[key],'circle',time=3.5,full=True,labels=False)
            axes[n-1,j].set_title(f'{n}人{n}车 · {m}\n{STATUS[row["status"]]} / 到达 {row["arrival_fraction"]:.0%}',fontsize=10)
    fig.suptitle('场景二完整对照｜六种数量 × 四种模型',fontsize=25,fontweight='bold')
    save(fig,out,'circle_models.png','同一行拥有完全相同的初态与主体参数，仅改变交互版本；含失败后的诊断轨迹。各图坐标与实体比例一致。')

    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for m in MODELS:
        rows=[metrics[f'circle_n{n}_{m}'] for n in range(1,7)];xx=np.arange(1,7)
        axes[0,0].plot(xx,[r['pedestrian_arrival_fraction']*100 for r in rows],marker='o',c=COLORS[m],label=m)
        axes[0,1].plot(xx,[r['car_arrival_fraction']*100 for r in rows],marker='o',c=COLORS[m],label=m)
        axes[1,0].plot(xx,[r['min_car_clearance_m'] for r in rows],marker='o',c=COLORS[m],label=m)
        axes[1,1].plot(xx,[np.nan if r['min_vehicle_clearance_m'] is None else r['min_vehicle_clearance_m'] for r in rows],marker='o',c=COLORS[m],label=m)
    for ax,title,ylabel in zip(axes.flat,['行人到达比例','车辆到达比例','人车最小身体净空','车车最小 SAT 分离裕度'],['%','%','m','m']):
        ax.set(title=title,xlabel='数量 N（N 人＋N 车）',ylabel=ylabel,xticks=range(1,7));ax.legend(ncol=2);ax.grid(alpha=.2)
    for ax in axes[1]:ax.axhline(0,c='black',ls=':',lw=.8)
    fig.suptitle('数量增加后的变化｜到达率与三类冲突分开评价',fontsize=21,fontweight='bold')
    save(fig,out,'circle_dynamics.png','GOAL 可能通过相互穿透到达；到达率必须与重叠状态合看。车车指标为 SAT 有符号分离裕度，不是矩形的精确欧氏距离。')

    fig,axes=plt.subplots(1,3,figsize=(17,8),layout='constrained')
    fields=['min_car_clearance_m','min_vehicle_clearance_m','min_pedestrian_clearance_m']
    for ax,field,title in zip(axes,fields,['人车表面净空','车车 SAT 分离裕度','人际表面净空']):
        values=np.array([[np.nan if metrics[f'circle_n{n}_{m}'][field] is None else metrics[f'circle_n{n}_{m}'][field] for m in MODELS] for n in range(1,7)])
        im=ax.imshow(np.ma.masked_invalid(values),cmap='RdYlGn',vmin=-1,vmax=1,aspect='auto')
        ax.set_xticks(range(4),MODELS);ax.set_yticks(range(6),[f'{n}人{n}车' for n in range(1,7)]);ax.set_title(title+'（m）')
        for i in range(6):
            for j in range(4):ax.text(j,i,'不适用' if np.isnan(values[i,j]) else f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('六组安全诊断｜人车、车车、人际全部检查',fontsize=23,fontweight='bold')
    save(fig,out,'circle_safety.png','1人1车无车车或人际配对，用不适用显示；任何一类发生重叠都计为失败。')

    fig,axes=plt.subplots(3,2,figsize=(15,11),layout='constrained',gridspec_kw={'height_ratios':[1,1,1.35]})
    for ax,m in zip(axes[:2].flat,MODELS):
        key=f'narrow_3.5_{m}';snapshot(ax,runs[key],agents[key],'narrow',time=6.8)
        ax.set_title(f'{m}｜4人＋1车｜{STATUS[metrics[key]["status"]]}',fontsize=11)
    for m in MODELS:
        r=runs[f'narrow_3.5_{m}'];v=np.linalg.norm(r['velocity'],axis=-1)
        axes[2,0].plot(r['time'],v[:,:4].mean(axis=1),c=COLORS[m],label=m);axes[2,1].plot(r['time'],v[:,4],c=COLORS[m],label=m)
    for ax,title in zip(axes[2],['四个行人平均速率','唯一汽车速率']):
        ax.set(xlim=(1,14),xlabel='时间（s）',ylabel='m/s',title=title);ax.legend(ncol=2);ax.grid(alpha=.2)
    fig.suptitle('场景一保留｜4 人＋1 车，3.5 m 直线窄路',fontsize=21,fontweight='bold')
    save(fig,out,'narrow_comparison.png','人车共同公式与参数不变；实体快照 t=6.8 s，横纵同比例。')

    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained');widths=[3.5,3.,2.8,2.4]
    for ax,field,title in zip(axes,['min_car_clearance_m','min_wall_clearance_m'],['人车最小净空','边界最小净空']):
        values=np.array([[metrics[f'narrow_{w:.1f}_{m}'][field] for m in MODELS] for w in widths]);im=ax.imshow(values,cmap='RdYlGn',vmin=-.5,vmax=.5,aspect='auto')
        ax.set_xticks(range(4),MODELS);ax.set_yticks(range(4),[f'{w:.1f} m' for w in widths]);ax.set_title(title+'（m）')
        for i,w in enumerate(widths):
            for j,m in enumerate(MODELS):ax.text(j,i,f'{values[i,j]:.3f}\n{STATUS[metrics[f"narrow_{w:.1f}_{m}"]["status"]]}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('场景一路宽对比｜人数始终为 4 人＋1 车',fontsize=21,fontweight='bold')
    save(fig,out,'width_comparison.png','车固定中心线，严格正净空并排通过至少需 W>2.8 m；几何可行不保证当前动力学规则安全。')

    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for n in [1,2,3,4,5,6]:
        r=runs[f'circle_n{n}_SIDE'];axes[0,0].plot(r['fine_time'],r['min_car_gap'],label=f'{n}人{n}车')
        if n>1:axes[0,1].plot(r['fine_time'],r['min_vehicle_gap'],label=f'{n}人{n}车')
    for ax,title in zip(axes[0],['SIDE 人车净空时序','SIDE 车车 SAT 分离裕度时序']):
        ax.axhline(0,c='black',ls=':');ax.set(xlim=(0,15),ylim=(-2,2),title=title,xlabel='时间（s）',ylabel='m');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
    for n,ax in zip([1,6],axes[1]):
        key=f'circle_n{n}_SIDE';snapshot(ax,runs[key],agents[key],'circle',time=3.5,full=True)
        ax.set_title(f'{n}人{n}车：{STATUS[metrics[key]["status"]]}')
    fig.suptitle('失败分析｜单对通行能力不能直接推广到多车交汇',fontsize=21,fontweight='bold')
    save(fig,out,'failure_analysis.png','车辆仅沿各自初始直线加减速，不能横向转向；软排斥和预测均不保证安全，冲突如实记录。')

    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for m in MODELS:
        rows=[next(s for s in steps if s['case']==f'circle_n{n}_{m}') for n in range(1,7)]
        axes[0].plot(range(1,7),[s['max_position_difference_m'] for s in rows],marker='o',c=COLORS[m],label=m)
        axes[1].plot(range(1,7),[abs(s['base_min_gap_m']-s['half_min_gap_m']) for s in rows],marker='o',c=COLORS[m],label=m)
    for ax,title in zip(axes,['共同采样点最大位置差','全程最小分离量变化绝对值']):
        ax.set(title=title,xlabel='数量 N（N人＋N车）',ylabel='m',xticks=range(1,7));ax.legend(ncol=2);ax.grid(alpha=.2)
    fig.suptitle('全部六组步长减半核查｜0.02 s → 0.01 s',fontsize=21,fontweight='bold')
    save(fig,out,'numerical_sensitivity.png',f'所有 {len(steps)} 项实验半步长重跑；状态改变 {sum(s["status_changed"] for s in steps)} 项。数值敏感性不是新数据上的验证。')

def make_readme(out,records,steps,robustness):
    def table(rows,keys,heads):
        def fmt(v):return '—' if v is None else (f'{v:.3f}' if isinstance(v,float) else str(v))
        return '| '+' | '.join(heads)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows)
    circle=[r for r in records if r['scenario']=='circle'];road=[r for r in records if r['scenario']=='narrow' and r['width_m']==3.5]
    ct=table(circle,['pedestrians','cars','model','status','min_car_clearance_m','min_vehicle_clearance_m','pedestrian_arrival_fraction','car_arrival_fraction'],['人数','车数','模型','状态','人车净空 m','车车 SAT 裕度 m','行人到达比例','车辆到达比例'])
    nt=table(road,['model','status','min_car_clearance_m','min_pedestrian_clearance_m','pass_time_s'],['模型','状态','人车净空 m','人际净空 m','全部错身 s'])
    rt=table(robustness,['scenario','pedestrians','cars','case','status','min_clearance_m'],['场景','人数','车数','扰动种子','状态','最小分离量 m'])
    text=f'''# Week 06：统一人车模型与六组圆周人车交会

**场景一保留 4 人＋1 车的 3.5 m 直线窄路；场景二完整实现 1人1车、2人2车、3人3车、4人4车、5人5车、6人6车六组。** 每组都运行 GOAL、DIST、PRED、SIDE 四种模型，共 24 个圆周对照；加上窄路四种宽度，共 40 个主实验。

每种模型内部，人和车共用目标驱动、交互与积分公式，尺寸、速度和自由度参数不同。此次补充了任意固定朝向的车辆矩形，以及车车交互和碰撞检测。多车不再全部当作沿全局 x 轴的单车。报告沿用 week5 的组织形式；仅修改 week6。

![六组总览](results/summary.png)

## 1. 场景、数据与范围

| 场景 | 行人数 | 车数 | 几何 |
|---|---:|---:|---|
| 场景一窄路 | 4 | 1 | 长40 m，主宽度3.5 m；另比较3.0、2.8、2.4 m |
| 场景二第1组 | 1 | 1 | 半径10 m，圆周对径目标 |
| 场景二第2组 | 2 | 2 | 同半径、同参数 |
| 场景二第3组 | 3 | 3 | 同半径、同参数 |
| 场景二第4组 | 4 | 4 | 同半径、同参数 |
| 场景二第5组 | 5 | 5 | 同半径、同参数 |
| 场景二第6组 | 6 | 6 | 同半径、同参数 |

圆周第 N 组：汽车起点角 theta_k=(k−(N−1)/2)·pi/N，k=0…N−1，在右半圆；对应行人位于汽车起点的相反点。所有 2N 个起点均匀分布在同一个圆上，各自目标为起点的负值。P1 对 C1，依次类推。没有圆形实体墙，也不是沿环形道路绕圈行驶。

每辆车沿自己的初始对径直线加减速，可停车、不倒车；在圆周实验中车辆朝向不相同，绘图与检测都使用真实朝向。没有增加自动转向或中途指定通行顺序。六组仅改变数量与由数量确定的均匀角度，半径和动力学参数保持固定。

![六组初始布局](results/circle_layouts.png)

全部为合成机制实验，初态可在 `data/scene_initial_conditions.json` 查看。当前不以旧的 64 人真实轨迹计算 ADE/FDE，不把它当成人车真实观测。历史版本仅在本地归档，未纳入本 GitHub 发布目录；当前 results 只包含最新的六组圆周交会与窄路结果。

## 2. 共同模型与参数

| 版本 | 共同机制 | 对照目的 |
|---|---|---|
| GOAL | 目标驱动＋道路边界，无主体避让 | 无交互基线 |
| DIST | GOAL＋当前净空排斥 | 距离反应 |
| PRED | DIST＋短时预测排斥 | 提前响应 |
| SIDE | PRED＋相对目标方向的右侧偏好 | 打破对称、产生绕行 |

这是三个递进的教学交互模型加一个无交互基线，不是四个独立论文算法的完整复现。公式详见 [模型说明](docs/model_equations.md)。共同参数仍为 A=5 m/s²、B=0.45 m、H=2 s、K=3.5 m/s²；边界 A=12 m/s²、B=0.15 m，没有按数量逐组调参。

| 参数 | 行人 | 汽车 |
|---|---:|---:|
| 身体 | 半径0.25 m圆 | 长4.4 m、宽1.8 m有向矩形 |
| 期望速度 | 1.5 m/s | 3 m/s |
| 响应时间尺度 | 0.5 s | 0.8 s |
| 加速度模长上限 | 3 m/s² | 4 m/s² |
| 自由度 | 平面运动 | 初始目标方向的直线运动 |

车辆将同一个加速度公式的输出投影到自身初始航向，速度同样投影并禁止倒车；不是给车另用跟驰模型。所有主体距离目标0.8 m内共同减速，进入0.25 m范围后停留并继续作为障碍。

## 3. 评价、车车几何与实验

圆—圆用欧氏表面净空，圆—有向矩形用局部坐标的精确有符号距离。**车—车采用分离轴定理 SAT：在两车的四个边法向上计算有符号分离裕度，取最大值。** 正值表示至少一个轴分离，负值表示所有轴重叠；其正值不是两矩形的精确欧氏最近距离，因此单独标注“车车 SAT 裕度”。所有几何都用于当前与预测交互，并用于实际重叠检测。

人车、人际、车车三类分别保存最小分离量，任一类重叠都判失败。`min_clearance_m` 是三类有符号分离量的最小值，只用于符号和安全诊断，不能当作统一的精确欧氏距离比较。单人单车不适用车车/人际指标，显示为空而非零。

成功要求全部主体到达、全程无重叠与越界。行人和车辆到达比例分别报告；`car_arrived` 表示所有车辆都到达。窄路错身要求全部行人越过唯一车尾；圆周错身要求每个行人沿其原直径方向完全越过自己的配对车尾。这个配对错身量是诊断，不能替代与其他车辆的安全检查。

40个主实验均运行40 s，内部dt=0.02 s、保存间隔0.1 s；全部40项用0.01 s重跑。窄路基础组和圆周六组的 SIDE 各做三个2 cm扰动种子，共21项。车辆初态扰动时目标同向平移，保持原固定航向。

发生重叠后仍继续的轨迹只作失效诊断，不把后续到达解读为安全成功；不强制拉开实体，不裁剪位置。首次失败按内部步记录，不是连续时间解析碰撞时刻。

## 4. 主要结果

### 4.1 场景一保留：4人1车、3.5 m窄路

{nt}

![窄路对照](results/narrow_comparison.png)

![宽度对照](results/width_comparison.png)

### 4.2 场景二六组 × 四模型完整结果

{ct}

![全部24个圆周对照](results/circle_models.png)

![人数车数增加后的指标](results/circle_dynamics.png)

![三类冲突热图](results/circle_safety.png)

不要只按到达率排序：无避让版本可能靠穿透完成目标；预测/绕行版本也可能因车车冲突、相持或行人绕行不当失败。车辆固定航向时，多条路线都穿过中心，单对通行能力不能直接推广为多车安全通行能力。

### 4.3 失败与数值敏感性

![失败分析](results/failure_analysis.png)

![步长检查](results/numerical_sensitivity.png)

全部40项步长减半后状态变化 **{sum(s['status_changed'] for s in steps)}** 项，共同采样点最大位置差 **{max(s['max_position_difference_m'] for s in steps):.3f} m**。状态不变并不表示路径已收敛：部分多车预测模型的轨迹差异较大，不能只凭失败类别相同就认定数值稳定。这是数值敏感性检查，不是不同真实实验上的验证。21项位置扰动结果如下：

{rt}

## 5. 判断与限制

当前重点是完整实现六组人车数量、用相同公式检查三类交互与失败，不能据此声称具有多车安全保证。模型没有驾驶转向、显式反应延迟、视线遮挡、硬接触动力学或通行优先权；车辆只能沿各自固定直线前进。软排斥加速度受限，仍可能重叠或停滞。右侧偏好是模型假设，没有真实人车数据标定或独立验证。

## 6. 输出与复现

本目录包含完整代码、9张结果图、同步动画、40项压缩轨迹与指标，可离线查看。GitHub上的HTML链接显示源文件；下载仓库后用浏览器打开 `results/replay.html` 即可播放，无需联网。历史归档、Python缓存和虚拟环境未上传。

```bash
python -m pip install -r requirements.txt
python run_experiments.py
python -m unittest discover -s tests -v
python validate_results.py
```

Python 3.10+，不依赖网络、外部数据或旧周代码；使用4个本地计算进程执行独立实验。`python reporting.py` 可从缓存结果重绘图表、动画与报告。

| 文件 | 内容 |
|---|---|
| [同步动画](results/replay.html) | 六组总览、六组分别选四模型、窄路切换、播放暂停、时间拖动 |
| `results/summary.png` | 六组SIDE结果总览 |
| `results/circle_layouts.png` | 六组初始位置、配对关系与车身朝向 |
| `results/circle_models.png` | 6×4完整圆周模型对照 |
| `results/circle_dynamics.png` | 数量与到达率、人车/车车分离量 |
| `results/circle_safety.png` | 人车、车车、人际三类热图 |
| `results/failure_analysis.png` | 六组时序与单对/多对失败对照 |
| `results/narrow_comparison.png` | 保留的窄路4人1车模型对比 |
| `results/width_comparison.png` | 窄路四种宽度 |
| `results/numerical_sensitivity.png` | 全部六组步长核查 |
| `results/experiment_metrics.csv` | 40项主实验完整记账 |
| `results/step_size_check.csv` | 40项半步长重复 |
| `results/robustness.csv` | 21项位置扰动 |
| `results/*.npz` | 所有主体的状态、尺寸及三类细步长分离量 |
| `data/scene_initial_conditions.json` | 每个实验的完整初态 |
| `results/parameters.json` | 固定参数与六组设置 |
| `results/validation.json` | 独立几何重算、人数车数与指标核查 |
| `results/reference_weeks_integrity.json` | 其他周只读完整性记录 |

## 7. 来源与范围

图片中的窄路要求作为场景一依据；之前圆周实验提供对径布局思路；week5提供交付组织参考。本版是在week6内自行实现的教学模型扩展，旧结果保留在归档，不混用旧人数、旧车数或旧误差指标。其他周不作修改。
'''
    (out.parent/'README.md').write_text(text,encoding='utf8')

def refresh_report():
    from prepare_data import ROOT
    from experiments import specifications
    from replay import make_replay
    def load(name):
        with (ROOT/'results'/name).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        for r in rows:
            for k,v in r.items():
                if k in {'scenario','case','model','status','base_status','half_status'}:continue
                if v=='':r[k]=None
                elif k in {'status_changed','car_arrived'}:r[k]=v=='True'
                elif k in {'agents','pedestrians','cars'}:r[k]=int(v)
                else:r[k]=float(v)
        return rows
    out=ROOT/'results';records=load('experiment_metrics.csv');steps=load('step_size_check.csv');runs={};agents={};metrics={}
    for s,row in zip(specifications(),records):
        key=s['key'];z=np.load(out/f'{key}.npz');runs[key]={k:z[k] for k in z.files};agents[key]=s['agents'];metrics[key]=row
    make_figures(out,runs,agents,metrics,steps);make_readme(out,records,steps,load('robustness.csv'));make_replay(out,runs,agents,metrics)

if __name__=='__main__':refresh_report()
