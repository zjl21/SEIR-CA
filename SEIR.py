import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

window_height = 500
window_width = 800
n_people = 5000
generation = 3000

exposed_rate_1m = 0.128
exposed_rate_3m = 0.026
distance_1m = 30
distance_3m = distance_1m * 3

incubation_period_min = 5
incubation_period_max = 14
incubation_period = np.zeros(n_people)
exposed_time = np.zeros(n_people)

recovery_period_min = 7
recovery_period_max = 14
recovery_period = np.zeros(n_people)
infected_time = np.zeros(n_people)

immunization_period_min = 60
immunization_period_max = 180
immunization_period = np.random.randint(immunization_period_min, immunization_period_max, size=n_people)
recovered_time = np.zeros(n_people)

death_rate = 0.01


x = np.random.choice(np.arange(window_width), size=n_people, replace=True)
y = np.random.choice(np.arange(window_height), size=n_people, replace=True)
speed = 5*np.ones(n_people)
status_sensitive = np.ones(n_people)
status_exposed = np.zeros(n_people)
status_infected = np.zeros(n_people)
status_recovered = np.zeros(n_people)



def move():
    global x, y, speed
    distance = np.random.rand(n_people)
    angle = np.random.rand(n_people)* 2 * np.pi
    x += (speed * distance * np.cos(angle, dtype=np.float16)).astype(np.int16)
    y += (speed * distance * np.sin(angle, dtype=np.float16)).astype(np.int16)

    x,y = np.maximum(x,0), np.maximum(y,0)
    x,y = np.minimum(x,window_width-1), np.minimum(y,window_height-1)

def calculate_probability(distance):
    if distance <= distance_1m:
        return exposed_rate_1m
    elif distance > distance_1m and distance < distance_3m:
        return exposed_rate_3m
    else:
        return 0
calc_prob = np.vectorize(calculate_probability, otypes=[np.float16])

def update_exposion():
    global status_sensitive, status_exposed, status_infected, status_recovered, status_death
    sensitive_idx = np.where(status_sensitive)[0]
    infected_idx = np.where(status_infected)[0]
    distance = np.sqrt((x[sensitive_idx] - x[infected_idx, np.newaxis]) ** 2 + (y[sensitive_idx] - y[infected_idx, np.newaxis]) ** 2)
   
    exposion_index = np.random.rand(*distance.shape)
    exposion_happen = exposion_index < calc_prob(distance)
    newly_exposed_idx = np.where(np.sum(exposion_happen, axis=0)>0)[0]
    newly_exposed = sensitive_idx[newly_exposed_idx]
    being_exposed(newly_exposed)


def change_status():
    global status_sensitive, status_exposed, status_infected, status_recovered, status_death
    exposed_idx = np.where(status_exposed == 1)
    infected_idx = np.where(status_infected == 1)
    recovered_idx = np.where(status_recovered == 1)

    death_index = np.random.rand(*infected_idx[0].shape)
    death_happen = death_index < death_rate
    newly_death = infected_idx[0][death_happen]
    being_death(newly_death)

    exposed_idx = np.where(status_exposed == 1)
    infected_idx = np.where(status_infected == 1)
    recovered_idx = np.where(status_recovered == 1)

    exposed_time[exposed_idx] += 1
    infected_time[infected_idx] += 1
    recovered_time[recovered_idx] += 1

    exposed_to_infected = exposed_idx[0][np.where(exposed_time[exposed_idx] > incubation_period[exposed_idx])[0]]
    infected_to_recovered = infected_idx[0][np.where(infected_time[infected_idx] > recovery_period[infected_idx])[0]]
    recovered_to_sensitive = recovered_idx[0][np.where(recovered_time[recovered_idx] > immunization_period[recovered_idx])[0]]

    being_infected(exposed_to_infected)
    being_recovered(infected_to_recovered)
    being_sensitive(recovered_to_sensitive)

def being_exposed(idx):
    global status_exposed, status_sensitive, incubation_period
    if len(idx) == 0:
        return
    status_exposed[idx] = 1
    status_sensitive[idx] = 0
    incubation_period[idx] = np.random.randint(incubation_period_min, incubation_period_max,size=len(idx))

def being_infected(idx):
    if len(idx) == 0:
        return
    status_infected[idx] = 1
    status_exposed[idx] = 0
    infected_time[idx] = np.random.randint(recovery_period_min, recovery_period_max,size=len(idx))

def being_recovered(idx):
    global status_recovered, status_infected, recovered_time
    if len(idx) == 0:
        return
    status_recovered[idx] = 1
    status_infected[idx] = 0
    recovered_time[idx] = np.random.randint(immunization_period_min, immunization_period_max,size=(1,len(idx)))

def being_sensitive(idx):
    global status_sensitive, status_recovered
    if len(idx) == 0:
        return
    status_sensitive[idx] = 1
    status_recovered[idx] = 0

def being_death(idx):
    global status_infected, status_exposed, status_recovered, status_sensitive, x, y, speed, incubation_period, exposed_time, recovery_period, infected_time, immunization_period, recovered_time, n_people
    if len(idx) == 0:
        return
    x = np.delete(x, idx)
    y = np.delete(y, idx)
    speed = np.delete(speed, idx)
    status_sensitive = np.delete(status_sensitive, idx)
    status_exposed = np.delete(status_exposed, idx)
    status_infected = np.delete(status_infected, idx)
    status_recovered = np.delete(status_recovered, idx)
    incubation_period = np.delete(incubation_period, idx)
    exposed_time = np.delete(exposed_time, idx)
    recovery_period = np.delete(recovery_period, idx)
    infected_time = np.delete(infected_time, idx)
    immunization_period = np.delete(immunization_period, idx)
    recovered_time = np.delete(recovered_time, idx)
    n_people -= len(idx)



def update():
    move()
    update_exposion()
    change_status()


def visualize():
    global status_sensitive, status_exposed, status_infected, status_recovered, status_death
    plt.clf()
    ax = plt.axes([0,0,1,1],facecolor='black')
    colors = np.empty_like(status_sensitive, dtype=object)
    colors[status_sensitive==1] = '#FFFFFF'
    colors[status_exposed==1] = '#FFC0CB'
    colors[status_infected==1] = '#FF0000'
    colors[status_recovered==1] = '#008000'

    plt.scatter(x,y,c=colors)
    plt.draw()
    plt.pause(0.001)



def main():
    infected_idx = np.random.randint(0,n_people)
    being_infected([infected_idx])
    fig = plt.figure(facecolor='black')
    plt.xlim(0, window_width)
    plt.ylim(0, window_height)
    for _ in tqdm(range(generation)):
        update()
        visualize()

main()