"""Report the fixed 21 training / 13 test quality-screened episodes."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reporting import configure,save,table,COLORS
from models import MODELS
from experiments import scenario


def render(episodes,data,audit,synthetic,real,detail,summary,convergence,out):
    configure();out=Path(out)
    selected=episodes[episodes.split=='test'].sort_values(['observed_speed_range_m_s','episode_id'],ascending=[False,True])
    chosen=selected.iloc[0];eid=chosen.episode_id;w=data[data.episode_id==eid]
    fig,axs=plt.subplots(3,3,figsize=(15,10),layout='constrained')
    for j,name in enumerate(('constant','braking','ngsim')):
        if name=='ngsim':
            tt=w.time_s.to_numpy()-w.time_s.iloc[0];vl=w.leader_v_m_s
            axs[0,j].plot(tt,w.observed_v_m_s,color='#263341',lw=2,label='后车参考')
            axs[1,j].plot(tt,w.gap_m,color='#263341',lw=2,label='参考间距')
            ss={m:real[eid,'calibrated',m] for m in MODELS};title='NGSIM 测试片段 · 训练标定参数'
        else:
            ar=scenario(name);tt=ar['t'];vl=ar['lv'];ss={m:synthetic[name,m] for m in MODELS}
            title=('前车匀速' if name=='constant' else '前车减速')+' · 默认参数'
        axs[0,j].plot(tt,vl,'--',color='#667380',label='前车')
        for m,sim in ss.items():
            for r,y in enumerate((sim.v,sim.gap,sim.a)):
                axs[r,j].plot(tt,y,color=COLORS[m],label=m,lw=1.5)
        axs[0,j].set_title(title);axs[0,j].legend(fontsize=8,ncol=2);axs[2,j].set_xlabel('时间（s）')
    for r,label in enumerate(('速度（m/s）','净间距（m）','加速度（m/s²）')):
        axs[r,0].set_ylabel(label)
    fig.suptitle('OVM / FVDM / IDM：速度响应、间距与加减速',fontsize=18,fontweight='bold')
    save(fig,out,'summary.png')

    test=summary[summary.split=='test']
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for j,(key,label) in enumerate([('position_rmse_m','完整片段位置RMSE均值（m）'),('speed_rmse_m_s','完整片段速度RMSE均值（m/s）'),('failure_rate','失败比例')]):
        for k,pname in enumerate(('default','calibrated')):
            vals=[test[(test.model==m)&(test.parameter_set==pname)][key].iloc[0] for m in MODELS]
            xx=np.arange(3)+(k-.5)*.34
            axs[0,j].bar(xx,vals,width=.32,color=[COLORS[m] for m in MODELS],alpha=.4 if k==0 else 1,hatch='//' if k==0 else None,label='默认' if k==0 else '标定')
            for x,y in zip(xx,vals):
                if np.isfinite(y):axs[0,j].text(x,y+max(max(vals)*.02,.005),f'{y:.3f}',ha='center',fontsize=8)
        axs[0,j].set_xticks(range(3),MODELS);axs[0,j].set_title(label,fontsize=11);axs[0,j].legend(fontsize=8)
        axs[0,j].set_ylim(0,max(axs[0,j].get_ylim()[1]*1.15,.08))
    dd=detail[(detail.split=='test')&(detail.parameter_set=='calibrated')]
    for j,(key,label) in enumerate([('position_rmse_m','逐片段位置RMSE（m）'),('speed_rmse_m_s','逐片段速度RMSE（m/s）'),('min_gap_m','逐片段最小净间距（m）')]):
        for k,m in enumerate(MODELS):
            vals=dd[dd.model==m][key].dropna().to_numpy()
            axs[1,j].scatter(k+np.linspace(-.15,.15,len(vals)),vals,color=COLORS[m],s=26)
            if len(vals):axs[1,j].plot([k-.22,k+.22],[np.median(vals)]*2,color='#263341',lw=2)
        axs[1,j].set_xticks(range(3),MODELS);axs[1,j].set_title(label,fontsize=11)
    fig.suptitle(f"NGSIM：{audit['selected_train']}段训练 / {audit['selected_test']}段测试，每段30秒\n训练与测试车辆无交叉；失败单列，误差不计初始点",fontsize=15,fontweight='bold')
    save(fig,out,'ngsim_comparison.png')

    # Show every test episode, so the representative panel cannot hide other cases.
    fig,axs=plt.subplots(4,4,figsize=(16,12),layout='constrained')
    for ax,(_,e) in zip(axs.flat,selected.iterrows()):
        ww=data[data.episode_id==e.episode_id];tt=ww.time_s.to_numpy()-ww.time_s.iloc[0]
        ax.plot(tt,ww.observed_v_m_s,color='#263341',lw=1.5,label='后车参考')
        ax.plot(tt,ww.leader_v_m_s,'--',color='#7f8b98',lw=1,label='前车')
        for m in MODELS:
            ss=real[e.episode_id,'calibrated',m];ax.plot(tt,ss.v,color=COLORS[m],lw=1.2,label=m)
            if ss.status!='success':ax.axvline(ss.failure_time_s,color=COLORS[m],ls=':')
        ax.set_title(f'L{int(e.lane_id)} F{int(e.follower_id)}/L{int(e.leader_id)}',fontsize=10)
        ax.set_xlabel('s');ax.set_ylabel('m/s')
    for ax in list(axs.flat)[len(selected):]:ax.set_axis_off()
    axs.flat[0].legend(fontsize=7,ncol=2)
    fig.suptitle('全部13段测试轨迹：标定参数自由滚动30秒，逐帧不重置',fontsize=16,fontweight='bold')
    save(fig,out,'ngsim_trajectories.png')

    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for j,(prefix,label) in enumerate((('observed','后车'),('leader','前车'))):
        tt=w.time_s.to_numpy()-w.time_s.iloc[0]
        axs[0,j].plot(tt,w[f'raw_{prefix}_v_m_s'],color='#aeb7c2',label='原始速度')
        axs[0,j].plot(tt,w[f'{prefix}_v_m_s'],color='#287fa0',label='位置样条导数')
        axs[0,j].set_title(label+'：代表测试片段');axs[0,j].set_ylabel('m/s');axs[0,j].legend()
    x=np.arange(len(episodes))
    axs[1,0].bar(x,episodes.raw_kinematic_rmse_m_s,label='原始',color='#d87828',alpha=.6)
    axs[1,0].bar(x,episodes.reconstructed_kinematic_rmse_m_s,label='处理后',color='#287fa0')
    axs[1,0].set(title='全部34段：位置/速度一致性RMSE',ylabel='m/s',xlabel='片段序号');axs[1,0].legend()
    axs[1,1].bar(x,episodes.position_adjustment_max_m,color='#7955a2')
    axs[1,1].axhline(.3,color='#d87828',ls=':',label='0.3 m门槛')
    axs[1,1].set(title='全部34段：最大位置去噪修正',ylabel='m',xlabel='片段序号');axs[1,1].legend()
    fig.suptitle('统一数据处理：原始质量筛选 → 小幅位置去噪 → 同源速度/加速度',fontsize=15,fontweight='bold')
    save(fig,out,'data_quality.png')

    fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    ar=scenario('emergency');axs[0].plot(ar['t'],ar['lv'],'--',color='#263341',label='前车')
    for m in MODELS:
        ss=synthetic['emergency',m]
        for r,y in enumerate((ss.v,ss.gap,ss.a)):
            axs[r].plot(ss.t,y,color=COLORS[m],label=m)
            if ss.status!='success':axs[r].axvline(ss.failure_time_s,color=COLORS[m],ls=':')
        if ss.status!='success':
            axs[1].scatter(ss.failure_time_s,0,marker='x',color=COLORS[m],s=70)
            axs[1].annotate(f'{m} 碰撞 {ss.failure_time_s:.2f} s',(ss.failure_time_s,0),xytext=(10,15),textcoords='offset points',color=COLORS[m])
    for r,label in enumerate(('速度（m/s）','净间距（m）','加速度（m/s²）')):axs[r].set_ylabel(label)
    axs[0].legend(ncol=4);axs[2].set_xlabel('时间（s）')
    fig.suptitle('补充压力测试：25 m/s、净间距15 m、前车5 s起以6 m/s²制动\n无加速度限幅；接触后停止',fontsize=14)
    save(fig,out,'failure_analysis.png')
    write_documents(episodes,data,audit,detail,summary,convergence,chosen,out)


def write_documents(episodes,data,audit,detail,summary,convergence,chosen,out):
    out=Path(out);root=Path(__file__).resolve().parent
    test=summary[summary.split=='test'];fit=test[test.parameter_set=='calibrated']
    bestx=fit.sort_values('position_rmse_m').iloc[0];bestv=fit.sort_values('speed_rmse_m_s').iloc[0]
    synth=pd.read_csv(out/'synthetic_metrics.csv');stress=synth[synth.scenario=='emergency']
    common=pd.read_csv(out/'ngsim_common_success_metrics.csv');common=common[(common.split=='test')&(common.parameter_set=='calibrated')]
    failures=detail[detail.status!='success']
    cfg=audit['config'];params=json.loads((out/'parameters.json').read_text())
    parameter_rows=[dict(model=m,**p) for m,p in params['calibrated'].items()]
    pd.DataFrame(parameter_rows).to_csv(out/'calibrated_parameters.csv',index=False)
    bound_hits=[]
    for m,p in params['calibrated'].items():
        for name,(low,high) in params['bounds'][m].items():
            if min(abs(p[name]-low),abs(p[name]-high))/(high-low)<.005:bound_hits.append(f'{m}.{name}={p[name]:.3f}')
    prior=root/'archive/initial_run/results/following_observations.csv'
    saved_audit=out/'previous_selection_audit.json'
    old_audit=json.loads(saved_audit.read_text(encoding='utf-8')) if saved_audit.exists() else {}
    if prior.exists():
        old=pd.read_csv(prior);peaks=[max(abs(np.diff(w.observed_v_m_s)/.1).max(),abs(np.diff(w.leader_v_m_s)/.1).max()) for _,w in old.groupby('episode_id')]
        old_audit=dict(episodes=len(peaks),accel_over_6=int(sum(p>6 for p in peaks)),maximum_implied_acceleration_m_s2=float(max(peaks)))
    (out/'previous_selection_audit.json').write_text(json.dumps(old_audit,indent=2),encoding='utf-8')
    lanes=episodes.groupby(['split','lane_id']).size().rename('episodes').reset_index()
    mismatch=convergence[convergence.status_dt05!=convergence.status_dt025]
    notes=f'''# OVM、FVDM、IDM：高质量NGSIM片段的复现与比较

本项目完成前车匀速、前车减速与NGSIM真实跟驰三组共同实验，另加紧急制动压力测试。真实实验固定使用已筛选的**21段训练、13段测试，每段30秒**；所有片段采用相同处理，训练/测试前后车编号无交叉。当前结果替代之前未经充分质量筛选的版本和单片段版本。仓库目录为 `week05/`，包含筛选数据、全部结果及压缩的原始公开数据，可独立复现。

![一页结果](results/summary.png)

## 1. 数据与范围

数据来自week4课程NGSIM I-80 SI文件，2005-04-13当地时间16:00–16:15。**week4保持只读，所有筛选、去噪和输出均在week5。** 本次采用较早时段内通过质量门槛的2–5号车道片段，不局限于week4的4号车道100米路段。

| 设置 | 内容 |
|---|---|
| 训练 | 21段，含缓冲的完整窗口结束≤250 s |
| 测试 | 13段，含缓冲的完整窗口开始≥300 s、结束≤500 s |
| 每段 | 30 s、301个0.1 s采样点，两端各5 s稳定跟驰缓冲 |
| 总评价时长 | 34×30=1020 s；训练630 s、测试390 s |
| 集合隔离 | 前车及后车编号交集为空；每位后车最多选1段 |
| 参数 | 理想实验用默认值；真实实验同时报告默认值与仅用21段训练标定的值 |

{table(lanes,['split','lane_id','episodes'],['集合','车道','片段数'])}

候选车道有2–5号，但测试样本主要落在2、3号车道，不能将测试结果称作所有车道均衡比较。筛选偏向稳定纵向跟驰，结论限于这一子集。

## 2. 为什么重新筛选，如何处理

NGSIM可以用于跟驰研究，问题是不能直接把未经检查的微观信号作为真实驾驶响应。上一版50段中37段存在相邻0.1秒速度差折算的加速度绝对值超过6 m/s²，极端值约158 m/s²。之前只凭同车道和前车不变，质量筛选不足。这个问题不能直接推翻week4的聚合基本图分析，二者关注的尺度不同。

本次在模型运行前固定以下观察数据门槛，不按拟合误差、碰撞或模型排名挑样本：

1. 同帧、同车道按纵向位置确定最近前车，先找前车再筛车型。源文件缺少前车编号，因此仍是几何推断。
2. 两车均为小汽车，车长3–6 m；前车编号与车道在40秒完整窗口内不变，连续0.1秒采样；评价中间30秒，前后各留5秒缓冲。
3. 原始净间距2–60 m；原始速度差分加速度绝对值≤6 m/s²；前后车位置—速度一致性RMSE均≤0.15 m/s，99%绝对误差≤0.6 m/s。
4. 去除固定道路曲率后，单车横向残差跨度≤1.5 m，两车横向残差差≤1.5 m。道路中心线由按纵向位置分箱的横向中位数拟合，仅描述固定几何，不拟合跟驰行为。
5. 后车均速≥2 m/s，移动时中位净时距≤4 s；至少一辆车速度变化范围≥2 m/s，保留动态信息。
6. 对40秒位置分别拟合三次平滑样条，残差预算`N×(0.1 m)²`；从同一样条求速度/加速度，最后去掉缓冲。不用三个跟驰模型重建观测数据。
7. 最大位置修正>0.3 m、速度修正RMSE>0.3 m/s、加速度>5 m/s²、jerk绝对值95%分位>5 m/s³或出现负速度时淘汰，不做强制截断。

共审计{audit['extended_candidate_windows']}个候选窗口，{audit['quality_pass_windows']}个通过质量筛选。候选允许重叠，不能当成独立样本。按原有车道/时间覆盖规则和每位后车最多一段选出固定21/13段；没有继续扩大数据搜索或为增加数量降低质量门槛。各阈值为本项目教学设置，完整配置见[data/selection_config.json](data/selection_config.json)。

所选片段原始位置—速度一致性RMSE最大{audit['selected_raw_kinematic_rmse_max_m_s']:.4f} m/s；统一处理后最大{audit['selected_reconstructed_kinematic_rmse_max_m_s']:.5f} m/s；最大位置修正{audit['selected_position_adjustment_max_m']:.3f} m，处理后最大加速度{audit['selected_reconstructed_accel_max_m_s2']:.3f} m/s²。原始列与处理列一起保存在`following_observations.csv`，每个候选的通过/拒绝原因见`candidate_quality.csv`。

本次使用的是**本地筛选并做小幅去噪的课程NGSIM**，不是作者发布的Montanino–Punzo重建数据，也不声称复现其完整重建算法。作者重建数据入口本次返回403，没有使用未取得的数据。数据质量处理和换道/短片段剔除有[研究依据](https://arxiv.org/abs/1610.03124)；不能据此声称这些本地处理轨迹就是无误差真值。

![数据质量与全部34段位置修正](results/data_quality.png)

## 3. 模型、共同实验与指标

净间距`s=x_L-x-L_L`，使用前车车长。OVM：`dv/dt=kappa[V(s)-v]`；FVDM另加`lambda(v_L-v)`；IDM使用自由加速与动态期望间距。OV曲线采用量纲化归一化tanh，IDM采用非负动态间距项；属于核心动力学教学复现，并非原论文逐图复制。公式、符号和单位见[模型说明](docs/model_equations.md)。

| 场景 | 输入 |
|---|---|
| 匀速 | 前车20 m/s、后车初速15 m/s、净间距35 m、60 s |
| 减速 | 双方初速20 m/s、净间距35 m，前车10–17.5 s以2 m/s²减至5 m/s，持续60 s |
| 真实跟驰 | 每段只用真实后车起点初始化，随后自由滚动30 s；前车回放，逐帧不重置后车 |
| 压力测试 | 双方初速25 m/s、净间距15 m，前车5 s起以6 m/s²制动到停，持续30 s |

默认自由速度均为30 m/s。OVM/FVDM共用`kappa=0.6 s⁻¹,h=25 m,ell=10 m`，FVDM另有`lambda=0.6 s⁻¹`；IDM：`a=1.2,b=2 m/s²,T=1.5 s,s0=2 m,delta=4`。相同初始条件不等于各模型都处于平衡状态。

真实实验仅用21段训练共同标定一个参数集，每模型两个固定初值、有界最小二乘。联合残差为位置误差/10 m和速度误差/2 m/s，不计初始点。13段测试不参与参数选择；原始与处理数据均保留，三个模型采用完全相同输入。参数及优化状态见[parameters.json](results/parameters.json)，表格见[calibrated_parameters.csv](results/calibrated_parameters.csv)。局部优化不保证全局最优；OVM有4个可调参数，FVDM/IDM各5个。

积分为弹道法，内部0.05 s，另以0.025 s核查。前车位置/速度在0.1秒采样点间分别线性插值，子步并非严格同一样条；处理后的采样点运动学一致性和积分敏感性分别核对。后车步内停止用精确停止距离，不倒车；不加速度限幅、不强制拉开间距，碰撞立即停止，后续NaN。

速度响应比较曲线、相对前车的速度RMSE和稳定时间（进入±0.5 m/s且直到结束不离开，减速从17.5 s算起）。真实RMSE先逐段计算，再对完整成功片段等权平均，初始点剔除。失败不计算完整RMSE，部分时长误差单列。净间距误差与后车位置误差只差符号，RMSE相同。`|a|>8 m/s²`单列强加减速诊断，不等同碰撞。

## 4. 主要结果

### 4.1 匀速与减速

{table(synth[synth.scenario.isin(['constant','braking'])],['scenario','model','status','settling_time_s','min_gap_m','speed_response_rmse_m_s'],['场景','模型','状态','稳定时间 s','最小净间距 m','速度响应RMSE m/s'])}

这组默认设置下FVDM稳定更快、OVM超调较明显，IDM匀速追赶较缓。FVDM匀速追赶初始加速度更大，速度差项也可能放大加速。IDM减速场景间距较小，不是所有工况都更保守。

### 4.2 13段真实测试

{table(test,['parameter_set','model','successes','collisions','numerical_failures','position_rmse_m','speed_rmse_m_s','minimum_gap_m'],['参数','模型','成功/13','碰撞','数值失败','位置RMSE m','速度RMSE m/s','最小净间距 m'])}

标定后位置误差最低为{bestx['model']}（{bestx['position_rmse_m']:.3f} m），速度误差最低为{bestv['model']}（{bestv['speed_rmse_m_s']:.3f} m/s）。该判断限于这13段、参数范围和联合损失；没有显著性检验，不视作普遍排名。

三个标定模型共同成功片段的同样本比较：

{table(common,['model','episodes','position_rmse_m','speed_rmse_m_s'],['模型','共同成功段数','位置RMSE m','速度RMSE m/s'])}

保持同一前车回放与同一预测，另对未经去噪的原始后车信号核对，避免只对平滑参考打分：

{table(fit,['model','raw_position_rmse_m','raw_speed_rmse_m_s'],['模型','对原始位置RMSE m','对原始速度RMSE m/s'])}

![默认与标定误差、失败和逐片段分布](results/ngsim_comparison.png)

![全部13段测试轨迹](results/ngsim_trajectories.png)

一页图代表片段按测试中实测后车速度变化幅度最大选取：`{chosen.episode_id}`；未按模型效果挑图。另展示全部13段测试轨迹，避免只看一个案例。

### 4.3 失败与数值敏感性

真实实验失败明细（空表表示未发生）：

{table(failures,['episode_id','split','parameter_set','model','status','failure_time_s'],['片段','集合','参数','模型','状态','失败时刻 s'])}

压力测试：

{table(stress,['model','status','failure_time_s','min_gap_m','max_abs_accel_m_s2'],['模型','状态','失败时刻 s','最小净间距 m','最大|a| m/s²'])}

![压力测试与首次接触](results/failure_analysis.png)

压力测试初态不是所有模型的平衡状态，初始强制动不能全解释为前车制动响应；没有现实制动限幅，未碰撞不等于现实安全保证。全部{len(convergence)}项步长减半核查中失败类型改变{len(mismatch)}项；共同有效时刻最大位置差{convergence.max_position_difference_m.max():.3f} m、速度差{convergence.max_speed_difference_m_s.max():.3f} m/s。逐项见[步长核查](results/step_size_check.csv)。

标定触边参数：{', '.join(bound_hits) if bound_hits else '无'}。这些参数可能辨识不足或相互补偿，不能直接当成驾驶员精确物理属性。误差目标也不包含独立安全裕度约束。

## 5. 最终判断与优缺点

| 模型 | 能反映的内容 | 优点 | 缺点 |
|---|---|---|---|
| OVM | 间距决定目标速度，车辆松弛调整 | 简单、快，易解释超调与振荡 | 不直接使用速度差，可能强加减速或碰撞 |
| FVDM | 间距反馈加双向速度差反馈 | 能直接响应相对运动，适合展示速度差信息的作用 | 多一个参数，线性项可放大加速度，不保证安全距离 |
| IDM | 自由加速、期望时距、静止间距与接近速度共同制动 | 参数含义直观，连接自由行驶与跟驰 | 非线性参数可能互相补偿，小间距可产生强制动，依赖数值与物理约束 |

**判断**：速度响应更快、轨迹误差更小和安全裕度更大是不同证据，需要同时看。此次改进重点是先剔除不适合纵向跟驰的噪声/换道附近片段，再作公平比较；不通过挑模型表现好的轨迹来得到结论。21/13段比单段更能看到个体差异，但仍属于一个时段的精选子集，不能验证交通流串稳定性、拥堵波、容量或换道行为。

## 6. 输出与复现

```bash
python -m pip install -r requirements.txt
python run_experiments.py
python -m unittest discover -s tests -v
python validate_results.py
```

需要Python 3.10或更新版本。在 `week05/` 中运行上述命令，脚本自动读取仓库附带的 `data/NGSIM_I80_20050413_1600-1615_SI.csv.gz`，无需解压或另外下载数据。也支持未压缩CSV或 `--data "完整CSV路径"`。完整运行包括固定规则选段、训练标定、全部评价、图表和README更新。原始数据只读，所有结果写入本目录。

| 文件 | 用途 |
|---|---|
| `results/episodes.csv` | 34段索引、前后车、车道、时间及质量指标 |
| `results/following_observations.csv` | 10,234点，原始和处理后信号同时保留 |
| `results/candidate_quality.csv` | 全部候选及逐段拒绝原因 |
| `results/ngsim_episode_metrics.csv` | 34段×两组参数×三个模型的完整记账 |
| `results/ngsim_summary.csv` | 训练/测试汇总 |
| `results/ngsim_predictions.csv` | 所有预测，失败后为NaN |
| `results/parameters.json` | 默认/标定值、边界与收敛记录 |
| `results/data_audit.json` | 来源哈希、筛选配置与统计 |
| `results/validation.json` | 独立误差重算与输出核查 |
| `results/week4_integrity.json` | week4运行前后文件哈希核对 |
| `data/NGSIM_I80_20050413_1600-1615_SI.csv.gz` | 完整课程SI源数据，gzip无损压缩，约26.24 MB |
| `data/package_manifest.json` | 压缩文件与解压内容的大小、SHA-256和来源 |

5张图分别为一页结果、数据质量、测试指标、全部测试轨迹和压力测试。GitHub目录沿用week4的组织方式，包含代码、图表、派生数据及压缩源数据；排除未压缩的大CSV、缓存与旧归档。`week4_integrity.json`记录本地实验时对原week4的核查；独立克隆没有相邻week4时不重复该本地核查。

## 7. 来源

数据：[U.S. DOT NGSIM](https://data.transportation.gov/d/8ect-6jqj)，课程SI文件SHA-256：`{audit['source_sha256']}`。

模型：[Bando et al., 1995](https://doi.org/10.1103/PhysRevE.51.1035)、[Jiang et al., 2001](https://doi.org/10.1103/PhysRevE.64.017101)、[Treiber et al., 2000](https://doi.org/10.1103/PhysRevE.62.1805)。筛选背景：[Kurtc & Treiber, 2016](https://arxiv.org/abs/1610.03124)；轨迹重建背景：[Montanino & Punzo, 2015](https://doi.org/10.1016/j.trb.2015.06.010)。本项目使用本地小幅去噪，不是后者发布的重建数据。
'''
    (root/'README.md').write_text(notes,encoding='utf-8')
    (out/'analysis_summary.md').write_text(notes.replace('](results/','](').replace('](docs/','](../docs/').replace('](data/','](../data/'),encoding='utf-8')
    (out/'run_manifest.json').write_text(json.dumps(dict(python=platform.python_version(),mode='fixed_21_train_13_test',
        selection_version=audit['selection_version'],source_sha256=audit['source_sha256'],randomness='none'),indent=2),encoding='utf-8')
