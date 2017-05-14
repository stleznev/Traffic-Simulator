#For helpful functions

import random as rd
import networkx as nx
from heapq import heapify, heappush, heappop


#Fucntion that finds index of the value in a two-level list (list of lists)
def find_index_2_levels(list_default, value):
	indexes = []
	for idx1 in range(len(list_default)):
		if value in list_default[idx1]:
			idx2 = list_default[idx1].index(value)
			indexes.append((idx1, idx2))
	return indexes


#Fucntion that finds given agent's subjective weight for the given edge at given time and given iteration
def calculate_subjective_weight_for_edge_time(graph, agent, time, edge, iteration, statistics):
	
	real_weight = graph[edge[0]][edge[1]]['weight']
	t = time
	indexes_edge = find_index_2_levels(statistics[iteration][agent.id_num].edges_passed, edge)

	#Agent never was on the edge
	if indexes_edge == []:
		subjective_weight = float(real_weight)
		#return subjective_weight
		return (subjective_weight, False)

	#If agent was on the edge
	if indexes_edge != []:
		for idx_edge in indexes_edge:
			time_interval_a = statistics[iteration][agent.id_num].real_edge_times[idx_edge[0]][idx_edge[1]][0]
			time_interval_b = statistics[iteration][agent.id_num].real_edge_times[idx_edge[0]][idx_edge[1]][1]
			
			#If agent previously arrived on the edge earlier or at the time = t
			if time >= time_interval_a and time <= time_interval_b:
		
				#Finding latest speed with which agent was going	
				for time_dict, speed_dict in statistics[iteration][agent.id_num].speed_by_time.items():
					if time_dict >= time_interval_a and time_dict <= time:
						current_speed = speed_dict
					if time_dict > time:
						break
					if time_dict < time_interval_a:
						continue

				#Get the list for adjustments of speed 		
				speed_preserver = [current_speed]
				time_preserver = [time]
				for time_dict, speed_dict in statistics[iteration][agent.id_num].speed_by_time.items():
					if time_dict > time and time_dict < time_interval_b:
						time_preserver.append(time_dict)
						speed_preserver.append(speed_dict)
				time_preserver.append(time_interval_b)
				speed_preserver.append(60)
				
				#Calculate objective and subjective weights agent has had before leaving the edge previously
				objective_weight = 0
				subjective_weight = 0
				for tp in range(1,len(time_preserver)):		
					objective_weight += (time_preserver[tp] - time_preserver[tp-1]) * speed_preserver[tp-1]
					subjective_weight += (time_preserver[tp] - time_preserver[tp-1]) * 60
				#Making weight adjusted for time left on it
				dist_left_to_go = (real_weight - objective_weight)
				objective_weight += dist_left_to_go
				subjective_weight += dist_left_to_go

				#return subjective_weight
				return (subjective_weight, True)

			if time > time_interval_b:
				continue

			if time < time_interval_a and (time + real_weight/60) <= time_interval_a:
				continue

			if time < time_interval_a and (time + real_weight/60) > time_interval_a:

				#Calculate distance agent would pass on edge by the time of first memories
				dist_passed = (time_interval_a - time) * 60
				dist_left_to_go = real_weight - dist_passed
				
				#Calculating current speed when agent suddenly remembers the speed on this edge
				for time_dict, speed_dict in statistics[iteration][agent.id_num].speed_by_time.items():
					if time_dict == time_interval_a:
						current_speed = speed_dict
						break

				#Get the list for adjustments of speed 		
				speed_preserver = [current_speed]
				time_preserver = [time_interval_a]
				for time_dict, speed_dict in statistics[iteration][agent.id_num].speed_by_time.items():
					if time_dict >= time_interval_a and time_dict < time_interval_b:
						time_preserver.append(time_dict)
						speed_preserver.append(speed_dict)
				time_preserver.append(time_interval_b)
				speed_preserver.append(60)

				#Calculate objective and subjective weights agent now has
				objective_weight = float(dist_passed)
				subjective_weight = float(dist_passed)
				for tp in range(1,len(time_preserver)):
					if objective_weight >= real_weight:
						break
					objective_weight += (time_preserver[tp] - time_preserver[tp-1]) * speed_preserver[tp-1]
					subjective_weight += (time_preserver[tp] - time_preserver[tp-1]) * 60
					saved_speed = speed_preserver[tp-1]
				
				weight_difference = objective_weight - real_weight
				if weight_difference == 0:
					#return subjective_weight
					return (subjective_weight, True)
				
				else:
					time_difference = weight_difference/saved_speed
					objective_weight -= weight_difference
					subjective_weight -= time_difference * 60
					
					#return subjective_weight
					return (subjective_weight, True)

	#return float(real_weight)
	return (float(real_weight), False)


# Include weighting of previous memories of edges
# Function that finds average subjective weight of the edge-time pair given all previous rides
def calculate_subjective_weight_for_edge_time_overall(graph, agent, time, edge, statistics):
	subjective_weights_all_iters = []
	discount = []
	for i in range(agent.current_iter+1):
		subjective_weight = calculate_subjective_weight_for_edge_time(graph = graph, 
																		agent = agent, 
																		time = time, 
																		edge = edge, 
																		iteration = i, 
																		statistics = statistics)
		if subjective_weight[1] != False:
			subjective_weights_all_iters.append(subjective_weight[0])

	# The final weight across all iterations is calculated by averaging all the estimates
	# of the previous iterations of simulation and real weight
	if subjective_weights_all_iters == []:
		return subjective_weight[0]
	else:
		discount = get_discount(number = len(subjective_weights_all_iters))
		weighted_average = 0
		for w in range(len(discount)):
			weighted_average += discount[w] * subjective_weights_all_iters[w]
		return weighted_average


# Function that finds the shortest path from source to target beginning in time = t0
def dijkstra_with_time_dependent_weights(graph, source, target, agent, time, statistics, length = False):
	
	digits_to_save = len(str(agent.time_stamp)) - 2
	
	adj_list = graph.adjacency_list()

	dist = [float('inf')]*len(adj_list)
	prev = [None]*len(adj_list)
	dist[source-1] = 0
	prev[source-1] = source-1

	dist_proxy = [(0,source-1)]
	init_time = float(time)
	
	while dist_proxy:
		min_val_node = heappop(dist_proxy)
		min_node = min_val_node[1]
		
		if min_node == target-1:
			if length == True:
				return dist[min_node]
			if length == False:
				return get_path(list_of_previous_nodes = prev, 
								target = target-1)
		
		if dist[min_node] != 0:
			time = float(("{0:." + str(digits_to_save) \
				+ "f}").format(round(init_time + (dist[min_node] / 60), digits_to_save)))
		
		for vertex in adj_list[min_node]:
			vertex = vertex - 1
			cost = calculate_subjective_weight_for_edge_time_overall(graph = graph, 
																		agent = agent,\
																		time = time, 
																		edge = (min_node+1, vertex+1), 
																		statistics = statistics)
			
			if dist[vertex] > dist[min_node] + cost:
				proxy_var = dist[vertex]
				dist[vertex] = dist[min_node] + cost
				heappush(dist_proxy, (dist[vertex], vertex))
				prev[vertex] = min_node


# Function that finds path from dijkstra algorithm
def get_path(list_of_previous_nodes, target):

	path = []
	v = list_of_previous_nodes[target]
	path.append(target+1)
	path.insert(0, v+1)
	
	while list_of_previous_nodes[v] != v:
		v = list_of_previous_nodes[v]
		path.insert(0, v+1)

	return path


# Function that outputs the weights for some number of periods
def get_discount(number, discount_rate = 0.1):
	discount = [0]
	div_element = 1
	for i in range(number):
		div_element -= discount[-1]
		divide_by = 0
		for n in range(number - i):
			divide_by += (1 + discount_rate) ** n
		discount.append(div_element / divide_by)
	return discount[1:]










