import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
# import matplotlib.animation as animation
# from moviepy.editor import ImageSequenceClip
import os
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ---------------------------------------- 调参部分开始 --------------------------------------------------------

window_height = 5000 # 地图高度（pixel）
window_width =  5000 # 地图宽度
n_people_total = 100000  # 总人数
generation = 365  # 迭代次数
gen_per_day = 1   # 一天有几次迭代
visible = False   # 是否显示动画，False就只输出结果的曲线，True就输出曲线+动画，输出动画的时间很长，建议调好曲线再开，
                  # 输出动画的过程中可以开另一个terminal继续调参尝试，但千万别忘了动画生成的是那一组参数的！

exposed_rate_1m = 0.608   # 1米内感染概率（12.8%）
exposed_rate_3m = 0.001   # 3米内感染概率（2.6%）
distance_1m = 1           # 1米对应的地图内的像素点距离

incubation_period_min = 1 * gen_per_day      # 潜伏期最小天数
incubation_period_max = 12 * gen_per_day    # 潜伏期最大天数

recovery_period_min = 1  * gen_per_day       # 康复期最小天数
recovery_period_max = 28 * gen_per_day       # 康复期最大天数

immunization_period_min = 7 * gen_per_day    # 免疫期最小天数
immunization_period_max = 365 * gen_per_day  # 免疫期最大天数

first_infected = 1000         # 初始感染人数

super_exposion_rate = 0.05  # 超级传播者出现概率
normal_speed  = 10000   # 一次迭代内人的移动最大距离
infected_speed = 1   # 感染者的移动速度

death_rate_0 = 0.0005      # 死亡率
# hospital_capacity = 0.1  # 医院容量
# delta_1 = 0.0005           # 死亡率增长率

isolation_distance = 3      # 密接距离，小于这个米数的会被隔离
isolation_rate = 0                 # 密接人群隔离率
isolation_rate_infected = 0 # 感染者隔离率
isolation_time = 14 * gen_per_day  # 隔离时间

herd_immunity_I_thr = 0.05     # 群体免疫阈值，感染人数占总人数比例达到这个值以下，认为达到群体免疫
herd_immunity_time_thr = 10 * gen_per_day     # 群体免疫天数阈值，多少天内感染人数占总人数比例小于阈值，认为达到群体免疫

result_name = 'SEIR_0611_test'     # 保存的结果图片名，记得及时存档，名称上最好表明参数的信息，以免忘记！
movie_name = 'SEIR_0610_not_isolate'      # 保存的视频名，记得及时存档，名称上最好表明参数的信息，以免忘记！
tmp_folder_name = 'tmp_not_isolate'            # 以免同时生成两个视频的时候混肴

# ---------------------------------------- 调参部分结束 --------------------------------------------------------

n_people = n_people_total  
distance_3m = distance_1m * 3
isolation_distance = distance_1m * isolation_distance
death_rate = death_rate_0
speed =  normal_speed * torch.ones(n_people_total,device=device)   

sensitive_all = torch.zeros(generation,device=device)
exposed_all = torch.zeros(generation,device=device)
infected_all = torch.zeros(generation,device=device)
recovered_all = torch.zeros(generation,device=device)
death_all = torch.zeros(generation,device=device)
isolate_all = torch.zeros(generation,device=device)

incubation_period = torch.zeros(n_people,device=device)
exposed_time = torch.zeros(n_people,device=device)

recovery_period = torch.zeros(n_people,device=device)
infected_time = torch.zeros(n_people,device=device)

immunization_period = torch.zeros(n_people,device=device)
recovered_time = torch.zeros(n_people,device=device)

isolated_time = torch.zeros(n_people,device=device)

x = torch.randint(0, window_width, (n_people,),device=device)
y = torch.randint(0, window_height, (n_people,),device=device)
status_sensitive = torch.ones(n_people,device=device)
status_exposed = torch.zeros(n_people,device=device)
status_infected = torch.zeros(n_people,device=device)
status_recovered = torch.zeros(n_people,device=device)
status_isolated = torch.zeros(n_people,device=device)


def move():
    global x, y, speed
    distance = torch.rand(n_people,device=device)
    angle = torch.rand(n_people,device=device)* 2 * np.pi
    x += (speed * distance * torch.cos(angle)).type(torch.int16)
    y += (speed * distance * torch.sin(angle)).type(torch.int16)

    x,y = torch.clamp_min(x,0), torch.clamp_min(y,0)
    x,y = torch.clamp_max(x,window_width-1), torch.clamp_max(y,window_height-1)

def calculate_probability(distance):
    prob = torch.zeros_like(distance,device=device)
    prob[distance <= distance_1m] = exposed_rate_1m
    prob[(distance > distance_1m) & (distance < distance_3m)] = exposed_rate_3m
    return prob

def calculate_isolation_probability(distance):
    prob = torch.zeros_like(distance,device=device)
    prob[distance <= isolation_distance] = isolation_rate
    return prob

def update_exposion():
    global status_sensitive, status_exposed, status_infected, status_recovered, status_death
    sensitive_idx = torch.nonzero(status_sensitive.bool() & ~status_isolated.bool()).squeeze()
    if sensitive_idx.size()==0:
        sensitive_idx.unsqueeze(0)

    infected_idx = torch.nonzero(status_infected.bool() & ~status_isolated.bool()).squeeze()

    # 隔离阳性
    isolate_idx = torch.rand_like(infected_idx.float())
    idx_isolate = infected_idx[isolate_idx < isolation_rate_infected]
    being_isolated(idx_isolate)

    infected_idx = torch.nonzero(status_infected.bool() & ~status_isolated.bool()).squeeze()

    distance = torch.sqrt((x[sensitive_idx] - x[infected_idx, None]) ** 2 + (y[sensitive_idx] - y[infected_idx, None]) ** 2)
   
    exposion_index = torch.rand(*distance.shape,device=device)
    # exposion_happen = torch.flatten(exposion_index < calculate_probability(distance))
    exposion_happen = exposion_index < calculate_probability(distance)
    if exposion_happen.shape == 2:
        exposion_happen = exposion_happen.squeeze(1)
    if len(exposion_happen.shape) == 1:
        exposion_happen = exposion_happen.unsqueeze(1)
    newly_exposed_idx = torch.nonzero(torch.sum(exposion_happen, axis=0)>0).squeeze()
    if len(sensitive_idx.shape) == 0:
        sensitive_idx = sensitive_idx.unsqueeze(0)
    newly_exposed = sensitive_idx[newly_exposed_idx]

    isolation_index = torch.rand(*distance.shape,device=device)
    isolation_happen = isolation_index < calculate_isolation_probability(distance)
    if isolation_happen.shape == 2:
        isolation_happen = isolation_happen.squeeze(1)
    if len(isolation_happen.shape) == 1:
        isolation_happen = isolation_happen.unsqueeze(1)
    newly_isolated_idx = torch.nonzero(torch.sum(isolation_happen, axis=0)>0).squeeze()
    newly_isolated = sensitive_idx[newly_isolated_idx]

    being_isolated(newly_isolated.squeeze()) # 隔离密接
    being_exposed(newly_exposed.squeeze())   # 感染

    # # 隔离阳性
    # isolate_idx = torch.rand_like(infected_idx.float())
    # idx_isolate = infected_idx[isolate_idx < isolation_rate_infected]
    # being_isolated(idx_isolate)


def change_status():
    global status_sensitive, status_exposed, status_infected, status_recovered, status_death
    infected_idx = torch.nonzero(status_infected == 1)

    death_index = torch.rand(*infected_idx.shape,device=device)
    death_happen = death_index < death_rate
    newly_death = infected_idx[death_happen]
    being_death(newly_death)

    exposed_idx = torch.nonzero(status_exposed == 1).squeeze()
    infected_idx = torch.nonzero(status_infected == 1).squeeze()
    recovered_idx = torch.nonzero(status_recovered == 1).squeeze()
    isolated_idx = torch.nonzero(status_isolated == 1).squeeze()

    exposed_time[exposed_idx] += 1
    infected_time[infected_idx] += 1
    recovered_time[recovered_idx] += 1
    isolated_time[isolated_idx] += 1

    if len(exposed_idx.shape) == 0:
        exposed_idx = exposed_idx.unsqueeze(0)
    if len(infected_idx.shape) == 0:
        infected_idx = infected_idx.unsqueeze(0)
    if len(recovered_idx.shape) == 0:
        recovered_idx = recovered_idx.unsqueeze(0)
    exposed_to_infected = exposed_idx[torch.nonzero(exposed_time[exposed_idx] > incubation_period[exposed_idx])]
    infected_to_recovered = infected_idx[(torch.nonzero(infected_time[infected_idx] > recovery_period[infected_idx]).squeeze())]
    recovered_to_sensitive = recovered_idx[torch.nonzero(recovered_time[recovered_idx] > immunization_period[recovered_idx])]

    being_infected(exposed_to_infected)
    being_recovered([infected_to_recovered])
    being_sensitive([recovered_to_sensitive])
    if len(isolated_idx.shape) == 0:
        isolated_idx = isolated_idx.unsqueeze(0)

    idx_disisolate = isolated_idx[torch.nonzero((isolated_time[isolated_idx] > isolation_time) & ~status_infected[isolated_idx].bool())] # 时间到了+没阳，解除隔离
    dis_isolation(idx_disisolate)

def being_exposed(idx):
    global status_exposed, status_sensitive, incubation_period
    if idx.numel() == 0:
        return
    status_exposed[idx] = 1
    status_sensitive[idx] = 0
    incubation_period[idx] = torch.randint(incubation_period_min, incubation_period_max,size=idx.shape,device=device).float()

def being_infected(idx):
    if len(idx) == 0:
        return
    idx = torch.tensor(idx)
    if len(idx.shape) > 1:
        idx = idx.squeeze()
    status_infected[idx] = 1
    status_exposed[idx] = 0
    exposed_time[idx] = 0
    recovery_period[idx] = torch.randint(recovery_period_min, recovery_period_max,size=idx.shape,device=device).float()
    dropout_idx = torch.rand_like(idx.float())
    idx_dropout = idx[dropout_idx >= super_exposion_rate]
    speed[idx_dropout] = infected_speed
    
    
def being_recovered(idx):
    global status_recovered, status_infected, recovered_time
    if len(idx) == 0:
        return
    status_recovered[idx] = 1
    status_infected[idx] = 0
    infected_time[idx] = 0
    immunization_period[idx] = torch.randint(immunization_period_min, immunization_period_max,size=(len(idx),)).float()
    speed[idx] = normal_speed

def being_sensitive(idx):
    global status_sensitive, status_recovered
    if len(idx) == 0:
        return
    status_sensitive[idx] = 1
    status_recovered[idx] = 0
    recovered_time[idx] = 0

def being_death(idx):
    global status_infected, status_exposed, status_recovered, status_sensitive, x, y, speed, incubation_period, exposed_time, recovery_period, infected_time, immunization_period, recovered_time, status_isolated, isolated_time, n_people
    if idx.numel() == 0:
        return
    mask = torch.ones(n_people, dtype=torch.bool)
    mask[idx] = 0
    x = x[mask]
    y = y[mask]
    speed = speed[mask]
    status_sensitive = status_sensitive[mask]
    status_exposed = status_exposed[mask]
    status_infected = status_infected[mask]
    status_recovered = status_recovered[mask]
    incubation_period = incubation_period[mask]
    exposed_time = exposed_time[mask]
    recovery_period = recovery_period[mask]
    infected_time = infected_time[mask]
    immunization_period = immunization_period[mask]
    recovered_time = recovered_time[mask]
    status_isolated = status_isolated[mask]
    isolated_time = isolated_time[mask]
    n_people -= len(idx)

def being_isolated(idx):
    global status_isolated
    if idx.numel() == 0:
        return
    status_isolated[idx] = 1
    isolated_time[idx] = 0
    speed[idx] = 0

def dis_isolation(idx):
    global status_isolated
    if idx.numel() == 0:
        return
    status_isolated[idx] = 0
    isolated_time[idx] = 0
    speed[idx] = normal_speed


def calc_death_rate():
    # if torch.sum(status_infected)/n_people_total <= hospital_capacity:
    #     death_rate = death_rate_0
    # else:
    #     death_rate = death_rate_0 + delta_1 * (torch.sum(status_infected)/n_people_total - hospital_capacity)
    # return torch.tensor(death_rate).to(device)
    return torch.tensor(death_rate_0).to(device)



def update():
    move()
    update_exposion()
    change_status()


def visualize(gen):
    global status_sensitive, status_exposed, status_infected, status_recovered, status_death
    plt.clf()
    ax = plt.axes([0,0,1,1],facecolor='black')

    idx_to_draw = torch.nonzero(~status_isolated.bool()).squeeze()
    sensitive_to_draw = torch.where(torch.isin(idx_to_draw, torch.nonzero(status_sensitive.bool() & ~status_isolated.bool()).squeeze()))[0]
    exposed_to_draw = torch.where(torch.isin(idx_to_draw, torch.nonzero(status_exposed.bool() & ~status_isolated.bool()).squeeze()))[0]
    infected_to_draw = torch.where(torch.isin(idx_to_draw, torch.nonzero(status_infected.bool() & ~status_isolated.bool()).squeeze()))[0]
    recovered_to_draw = torch.where(torch.isin(idx_to_draw, torch.nonzero(status_recovered.bool() & ~status_isolated.bool()).squeeze()))[0]

    colors = np.empty_like(idx_to_draw.cpu(), dtype=object)
    colors[sensitive_to_draw.cpu().numpy()] = '#FFFFFF'
    colors[exposed_to_draw.cpu().numpy()] = '#FFC0CB'
    colors[infected_to_draw.cpu().numpy()] = '#FF0000'
    colors[recovered_to_draw.cpu().numpy()] = '#008000'

    plt.scatter(x[idx_to_draw].cpu().numpy(),y[idx_to_draw].cpu().numpy(),c=colors,s=1)
    # plt.draw()
    # plt.pause(0.001)
    plt.savefig(f'{tmp_folder_name}/frame_{gen}.png')



def main():
    global death_rate
    herd_immunity_time = None
    infected_idx = torch.randint(0,n_people, (first_infected,))
    being_infected(infected_idx)
    if visible:
        os.makedirs(f'{tmp_folder_name}',exist_ok=True)
        fig = plt.figure(facecolor='black')
        plt.xlim(0, window_width)
        plt.ylim(0, window_height)
    for gen in tqdm(range(generation)):
        death_rate = calc_death_rate()
        update()
        sensitive_all[gen] = torch.sum(status_sensitive)
        exposed_all[gen] = torch.sum(status_exposed)
        infected_all[gen] = torch.sum(status_infected)
        recovered_all[gen] = torch.sum(status_recovered)
        death_all[gen] = n_people_total - n_people
        isolate_all[gen] = torch.sum(status_isolated)
        if (infected_all[-herd_immunity_time_thr:] < n_people_total * herd_immunity_I_thr).all() and (herd_immunity_time is None) and gen > herd_immunity_time_thr:
            herd_immunity_time = gen
        if infected_all[gen] > n_people_total * herd_immunity_I_thr:
            herd_immunity_time = None
        if visible:
            visualize(gen)

    # 最后把四条曲线画出来，保存到文件夹里
    plt.figure()
    plt.plot(np.arange(0,generation)/gen_per_day, sensitive_all.cpu().numpy()/n_people_total, label='sensitive')
    plt.plot(np.arange(0,generation)/gen_per_day, exposed_all.cpu().numpy()/n_people_total, label='exposed',color='pink')
    plt.plot(np.arange(0,generation)/gen_per_day, infected_all.cpu().numpy()/n_people_total, label='infected',color='red')
    plt.plot(np.arange(0,generation)/gen_per_day, recovered_all.cpu().numpy()/n_people_total, label='recovered',color='green')
    plt.plot(np.arange(0,generation)/gen_per_day, death_all.cpu().numpy()/n_people_total, label='death',color='black')
    plt.plot(np.arange(0,generation)/gen_per_day, isolate_all.cpu().numpy()/n_people_total, label='isolated',color='yellow')

    death_last = (n_people_total - n_people)
    txt = f"death rate: {death_last:.4f}"

    plt.legend()
    plt.xlabel('Time (Days)')
    plt.ylabel('Proportion')
    # plt.text(generation/gen_per_day*0.8, -0.03, txt, fontsize=12, ha='center')
    plt.savefig(f'{result_name}.png')
    # plt.show()

    print(f"death rate: {death_last:.5f}")
    if herd_immunity_time is None:
        print("herd immunity not reached")
    else:
        print(f"herd immunity time: {herd_immunity_time/gen_per_day:.2f} days")

main()


if visible:
    filenames = [f'{tmp_folder_name}/frame_{i}.png' for i in range(generation)] 
    clip = ImageSequenceClip(filenames, fps=24)  # fps 是每秒的帧数，你可以根据需要修改
    clip.write_videofile(f'{movie_name}.mp4')
    shutil.rmtree(f'{tmp_folder_name}')