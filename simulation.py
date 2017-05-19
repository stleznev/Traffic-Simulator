import time
from collections import OrderedDict
from copy import deepcopy
from heapq import heapify, heappush, heappop

from sklearn.ensemble import AdaBoostRegressor
import networkx as nx
import numpy as np

from baseline_dijkstra import find_index_2_levels, get_path, get_discount
from baseline_dijkstra import calculate_subjective_weight_for_edge_time_overall
from baseline_dijkstra import dijkstra_with_time_dependent_weights


class Simulation(object):
	def __init__(self, graph, agents, max_time, number_of_iterations = 1):

		#Parameters initialization
		self.max_time = max_time
		self.graph = graph
		self.agents = agents
		self.statistics = dict()
		self.iteration = 0
		self.number_of_iterations = number_of_iterations
		self.paths_changed =dict()


	#Function that counts how speed on the road is affected by the number of cars on it
	def speed_revisited(self, number_of_cars_before_agent_on_an_edge, agent):
		agent.speed = 60
		if number_of_cars_before_agent_on_an_edge <= 2:
			agent.speed = agent.speed
		elif number_of_cars_before_agent_on_an_edge <= 4 and \
		number_of_cars_before_agent_on_an_edge > 2:
			agent.speed = agent.speed - 10
		elif number_of_cars_before_agent_on_an_edge <= 6 and \
		number_of_cars_before_agent_on_an_edge > 4:
			agent.speed = agent.speed - 20
		elif number_of_cars_before_agent_on_an_edge <= 8 and \
		number_of_cars_before_agent_on_an_edge > 6:
			agent.speed = agent.speed - 30
		elif number_of_cars_before_agent_on_an_edge <= 10 and \
		number_of_cars_before_agent_on_an_edge > 8:
			agent.speed = agent.speed - 30
		elif number_of_cars_before_agent_on_an_edge <= 12 and \
		number_of_cars_before_agent_on_an_edge > 10:
			agent.speed = agent.speed - 40
		elif number_of_cars_before_agent_on_an_edge > 12:
			agent.speed = agent.speed - 50
		return int(agent.speed)


	#Function that calculates the distance agent passed on the current edge 
	#by current time
	def calculate_passed_distance(self, agent, current_time):
		time_passed_on_edge = [0]
		passed_distance = 0
		speed_preserver = [0]
		agent.speed_by_time = OrderedDict(sorted(agent.speed_by_time.items()))
		for time, speed in agent.speed_by_time.items():
			if time >= agent.real_edge_times[agent.current_path][agent.current_edge_num][0] \
			and time <= current_time:
				time_passed_on_edge.append(time - agent.real_edge_times[agent.current_path][agent.current_edge_num][0])
				speed_preserver.append(speed)
				current_part_passed_time = time_passed_on_edge[-1] - time_passed_on_edge[-2]
				passed_distance += current_part_passed_time * speed_preserver[-2]
			if time > current_time:
				break
		return float(("{0:." + str(len(str(self.agents[0].time_stamp)) - 2) + "f}").\
			format(round(passed_distance, len(str(self.agents[0].time_stamp)) - 2)))


	#Function that sorts speed_by_time variable of agents by time
	def sort_speed_by_time_for_all(self):
		for a in self.agents:
			a.speed_by_time = OrderedDict(sorted(a.speed_by_time.items()))


	#Fucntion that calculates real edge times for path->edge
	def calculate_real_edge_time_for_edge(self, agent, path_num, edge_num):
		digits_to_save = len(str(agent.time_stamp)) - 2
		if edge_num == 0:
			#Real edge time is calculated at first by dividing objective 
			#weight of the edge by agent's real speed at the moment he arrived
			#on it, e.g. agent makes a forecast in which time he will finish
			#the edge with current speed
			time = agent.times_of_departures[path_num]
			edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
			agent.real_edge_times[path_num][edge_num] = \
			list((time, time + (self.graph[edge[0]][edge[1]]['weight'] / agent.speed)))

		else:
			time = agent.real_edge_times[path_num][edge_num-1][1]
			edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
			agent.real_edge_times[path_num][edge_num] = \
			list((time, time + (self.graph[edge[0]][edge[1]]['weight'] / agent.speed)))
		#Rounding to preserve discrete time consistency
		for fl in range(2):
			agent.real_edge_times[path_num][edge_num][fl] = \
			float(("{0:." + str(digits_to_save) + "f}").format(round(agent.real_edge_times[path_num][edge_num][fl], digits_to_save)))


	#Fucntion that copies agents and saves statistics about them
	def copy_agents(self):
		copied_agents = []
		for a in self.agents:
			copied_agents.append(deepcopy(a))
		return copied_agents


	#Function that restores every agent's parameters to initial 
	#ones (uses function agent.restore()
	#from class Agent)
	def restore_all_agents(self):
		for a in self.agents:
			a.restore()


	#Function that restores every agent's parameters to initial ones, 
	#also restoring current_iteration parameter
	def restore_all_agents_finally(self):
		for a in self.agents:
			a.restore(finish = True)


	#Function that searches across all agents and identifies ones with 
	#incomplete paths. After such identification function assigns new
	#maximim time for the simulation in order for all drivers to finish their trips
	def check_agents_prolong_simulation(self):
		for a in self.agents:
			for path_num in range(len(a.paths)):
				if a.paths[path_num] == None:
					if self.max_time < a.times_of_departures[path_num]:
						self.max_time = a.times_of_departures[path_num]
						#print('Changed max_time', self.max_time)
					else:
						pass
				else:
					for edge_num in range(len(a.paths[path_num]) - 1):
						if a.real_edge_times[path_num][edge_num][-1] != None \
						and a.real_edge_times[path_num][edge_num][-1] > self.max_time:
							self.max_time = a.real_edge_times[path_num][edge_num][-1]
							break





#Class that defines the simulation of baseline traffic situation,
#agents go according to the shortest-distance paths with weights 
#they memorized during previous iterations
#and also current iteration (however this makes no effect on the weights 
#because agents memorize only congestion-time weight, not just congestion).
class simulation_baseline(object):
	def __init__(self, graph, agents, max_time, number_of_iterations = 1):
		
		#Parameters initialization
		self.max_time = max_time
		self.graph = graph
		self.agents = agents
		self.statistics = dict()
		self.common_sense = dict()
		self.iteration = 0
		self.number_of_iterations = number_of_iterations
		self.paths_changed =dict()


	#Function that counts how speed on the road is affected by the number of cars on it
	def speed_revisited(self, number_of_cars_before_agent_on_an_edge, agent):
		agent.speed = 60
		if number_of_cars_before_agent_on_an_edge <= 2:
			agent.speed = agent.speed
		elif number_of_cars_before_agent_on_an_edge <= 4 and \
		number_of_cars_before_agent_on_an_edge > 2:
			agent.speed = agent.speed - 10
		elif number_of_cars_before_agent_on_an_edge <= 6 and \
		number_of_cars_before_agent_on_an_edge > 4:
			agent.speed = agent.speed - 20
		elif number_of_cars_before_agent_on_an_edge <= 8 and \
		number_of_cars_before_agent_on_an_edge > 6:
			agent.speed = agent.speed - 30
		elif number_of_cars_before_agent_on_an_edge <= 10 and \
		number_of_cars_before_agent_on_an_edge > 8:
			agent.speed = agent.speed - 30
		elif number_of_cars_before_agent_on_an_edge <= 12 and \
		number_of_cars_before_agent_on_an_edge > 10:
			agent.speed = agent.speed - 40
		elif number_of_cars_before_agent_on_an_edge > 12:
			agent.speed = agent.speed - 50
		return int(agent.speed)


	#Function that calculates the distance agent passed on the current edge 
	#by current time
	def calculate_passed_distance(self, agent, current_time):
		time_passed_on_edge = [0]
		passed_distance = 0
		speed_preserver = [0]
		agent.speed_by_time = OrderedDict(sorted(agent.speed_by_time.items()))
		for time, speed in agent.speed_by_time.items():
			if time >= agent.real_edge_times[agent.current_path][agent.current_edge_num][0] \
			and time <= current_time:
				time_passed_on_edge.append(time - agent.real_edge_times[agent.current_path][agent.current_edge_num][0])
				speed_preserver.append(speed)
				current_part_passed_time = time_passed_on_edge[-1] - time_passed_on_edge[-2]
				passed_distance += current_part_passed_time * speed_preserver[-2]
			if time > current_time:
				break
		return float(("{0:." + str(len(str(self.agents[0].time_stamp)) - 2) + "f}").\
			format(round(passed_distance, len(str(self.agents[0].time_stamp)) - 2)))


	#Function that sorts speed_by_time variable of agents by time
	def sort_speed_by_time_for_all(self):
		for a in self.agents:
			a.speed_by_time = OrderedDict(sorted(a.speed_by_time.items()))


	#Function that calculates expected edge times for path->edge
	def calculate_expected_edge_time_for_edge(self, agent, path_num, edge_num):
		digits_to_save = len(str(agent.time_stamp)) - 2
		if edge_num == 0:
			#Expected edge time is calculated as the edge subjective (for agent) weight 
			#divided by 60 kmh (default speed). Expected edge time is a time in which 
			#agent expects to finish traversing the edge
			time = agent.times_of_departures[path_num]
			edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
			agent.expected_edge_times[path_num].append(list((time, time \
			+ (calculate_subjective_weight_for_edge_time_overall(graph = self.graph, \
																		agent = agent, 
																		time = time,
																		edge = edge,
																		statistics = self.common_sense) \
			/ agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			agent.real_edge_times[path_num].append(list((time, None)))
		
		else:
			time = agent.expected_edge_times[path_num][edge_num-1][1]
			edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
			agent.expected_edge_times[path_num].append(list((time, time \
			 + (calculate_subjective_weight_for_edge_time_overall(graph = self.graph, \
																	agent = agent, 
																	time = time,
																	edge = edge,
																	statistics = self.common_sense) \
			 / agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			agent.real_edge_times[path_num].append(list((None, None)))
		#Everything is rounded to maintain discrete time consistency
		for fl in range(2):
			agent.expected_edge_times[path_num][edge_num][fl] = \
			float(("{0:." + str(digits_to_save) + "f}").format(round(agent.expected_edge_times[path_num][edge_num][fl], digits_to_save)))
		agent.real_edge_times[0][0][0] = \
		float(("{0:." + str(digits_to_save) + "f}").format(round(agent.real_edge_times[0][0][0], digits_to_save)))


	#Fucntion that calculates real edge times for path->edge
	def calculate_real_edge_time_for_edge(self, agent, path_num, edge_num):
		digits_to_save = len(str(agent.time_stamp)) - 2
		if edge_num == 0:
			#Real edge time is calculated at first by dividing objective 
			#weight of the edge by agent's real speed at the moment he arrived
			#on it, e.g. agent makes a forecast in which time he will finish
			#the edge with current speed
			time = agent.times_of_departures[path_num]
			edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
			agent.real_edge_times[path_num][edge_num] = \
			list((time, time + (self.graph[edge[0]][edge[1]]['weight'] / agent.speed)))

		else:
			time = agent.real_edge_times[path_num][edge_num-1][1]
			edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
			agent.real_edge_times[path_num][edge_num] = \
			list((time, time + (self.graph[edge[0]][edge[1]]['weight'] / agent.speed)))
		#Rounding to preserve discrete time consistency
		for fl in range(2):
			agent.real_edge_times[path_num][edge_num][fl] = \
			float(("{0:." + str(digits_to_save) + "f}").format(round(agent.real_edge_times[path_num][edge_num][fl], digits_to_save)))


	#LAST CHANGES:
	#2) cntr instead of index(agent)
	#3) Direct path adressing for agent's path instead of agent.class_method()
	#Function which makes 1 iteration of global simulation loop
	def simulation_iteration(self):

		digits_to_save = len(str(self.agents[0].time_stamp)) - 2

		#Priorty qeues for every edge and counters for each edge to check whether every agent who
		#came to edge also left it
		agents_on_edges = dict((e,[]) for e in self.graph.edges())
		edge_name_came = dict((e, 0) for e in self.graph.edges())
		edge_name_left = dict((e, 0) for e in self.graph.edges())

		#Main Loop that makes each iteration of the loop tick with the time interval specified by time_stamp
		#for i in range(0,int((self.max_time/self.agents[0].time_stamp))):
			#t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
				#	"f}").format(round((0 + i*self.agents[0].time_stamp), digits_to_save)))
		arch_time = -1
		while arch_time <= int((self.max_time/self.agents[0].time_stamp)):
			arch_time += 1
			t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
				"f}").format(round((0 + arch_time*self.agents[0].time_stamp), digits_to_save)))

			#Agent leaves the edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num])-1):

							#Agent leaves the edge
							if a.real_edge_times[path_num][edge_num][1] == t:

								#Find the edge from which agent leaves and decrease the number of cars on it by one
								#Also, dequeue agent from this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])

								#print('---------GUYS ON EDGE----------')
								#print('EDGE', edge_name)
								#for agn in agents_on_edges[edge_name]:
								#	print('ID guy', (agn.id_num))
								
								index = agents_on_edges[edge_name].index(a)
								agents_on_edges[edge_name].pop(index)
								#if index != 0:
								#	print('Time', t, 'Agent', a.id_num, 'IDX', index)
								#	agents_on_edges[edge_name].pop(index)
								#if index == 0:
								#	agents_on_edges[edge_name].pop(0)

								#Check
								edge_name_left[edge_name] += 1

								#Checking prints
								#print('----------LEAVING----------')
								#print('agent id', a.id_num)
								#print('number_of_cars_after', len(agents_on_edges[edge_name]))
								#print('time',t)
								#print('edge',edge_name)

								#Cleans current edge variables and transfers time of edge-finish to next edge start time
								#Or in case this is the end the path for agent increases his current path number by one
								a.current_edge = (0,0)
								a.current_edge_num = []
								if edge_num <= (len(a.real_edge_times[path_num])-2):
									a.real_edge_times[path_num][edge_num+1][0] = a.real_edge_times[path_num][edge_num][1]
								if edge_num == len(a.real_edge_times[path_num])-1:
									a.current_path += 1

								#Recalculating real travel times for every agent on an edge 
								#(who is not leacing at the current time) from which another agent left
								#Their speed must change due to decrese in number of cars ahead of them
								cntr = -1
								for agent in agents_on_edges[edge_name]:
									cntr += 1
									if agent.current_edge == edge_name and \
									(agent.real_edge_times[agent.current_path][agent.current_edge_num][1] != t):

										#Searching for the position of the agent on an edge and finding number of cars ahead
										idx_of_agent = agents_on_edges[edge_name].index(agent)
										if (int(cntr) != idx_of_agent):
											print('Time', t, 'agent', agent.id_num)
										#num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name][:idx_of_agent])
										num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name][:int(cntr)])

										#Recalculating speed and saving it in an ordered dict
										agent.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, agent = agent)
										agent.speed_by_time[t] = int(agent.speed)

										#Calculating the distance on edge which is already passed by an agent, distance to pass,
										#and finally renovating the real travel time on the edge
										passed_distance = self.calculate_passed_distance(agent = agent, 
																							current_time = t)
										weight_of_the_edge = self.graph[edge_name[0]][edge_name[1]]['weight']
										distance_to_pass = float(("{0:." + str(digits_to_save) + "f}").\
										format(round(weight_of_the_edge - passed_distance, \
											digits_to_save)))

										#Checking prints
										#print('----------------')
										#print('Time', t)
										#print('agent id', agent.id_num)
										#print('Path', agent.current_path)
										#print("Current edge", agent.current_edge)
										#print('Time of arrival before change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Passed Distance', passed_distance)

										time_to_pass = float(("{0:." + str(digits_to_save) + "f}").\
											format(round(distance_to_pass / agent.speed, 2)))
										#Small fix to secure problems with rounding
										if time_to_pass == 0.0:
											time_to_pass = agent.time_stamp
										agent.real_edge_times[agent.current_path][agent.current_edge_num][1] = \
										float(("{0:." + str(digits_to_save) + "f}").format(round(t + time_to_pass, digits_to_save)))

										#Checking prints continued
										#print('Time of arrival after change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Speed after', agent.speed)
										#print('NUM_BOYS', num_of_cars_before_agent_on_an_edge)
								#Break edge_num loop
								break

			#Delete variables from memory
			#del a
			#del path_num

			#Agent finds path and expects to traverse it for some time Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					
					#If it is time to go - agent finds the path from source to target
					if a.times_of_departures[path_num] == t and \
					a.paths[path_num] == None and a.current_edge == (0,0):
						a.paths[path_num] = \
							dijkstra_with_time_dependent_weights(	graph = self.graph, 
																	source = a.start_nodes[path_num],
																	target = a.finish_nodes[path_num], 
																	agent = a, 
																	time = t, 
																	statistics = self.common_sense, 
																	length = False	)
						
						#To maintain expectation correctness we need to increase agent's speed to 60 kmh
						a.speed = 60
						#Then he finds the expected time he will be on each edge, also, expected travel time
						for edge_num in range(len(a.paths[path_num])-1):
							self.calculate_expected_edge_time_for_edge(agent = a,
																		path_num = path_num, 
																		edge_num = edge_num)
						#Refuse trips if the expected time of current trip
						#is more than maximum simulation time - some constant
						#a.refuse_trips(path_num = path_num, max_time = self.max_time - 2)
						#if len(a.paths) <= path_num:
						#	print('REFUSED')
						#	break

						if path_num <= len(a.paths)-2:
							for i in range(1,len(a.paths)-path_num):
								if a.expected_edge_times[path_num][-1][-1] >= a.times_of_departures[path_num+i]:
									a.times_of_departures[path_num+i] = float(a.expected_edge_times[path_num][-1][-1]) + 0.01*i
									a.times_of_departures[path_num+i] = \
									float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + "f}")\
										.format(round(a.times_of_departures[path_num+i], digits_to_save)))


			#Delete variables from memory
			#del a
			#del path_num

			#Agent comes to edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num])-1):
							
							#Agent comes to edge
							if a.real_edge_times[path_num][edge_num][0] == t and \
							a.current_edge == (0,0) and a.current_path == path_num:

								#Find the edge on which agent comes and increase the number of cars on it by one
								#Also, enqueue the agent on this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])
								num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name])
								agents_on_edges[edge_name].append(a)

								#Increase the daily number of cars came to the edge
								edge_name_came[edge_name] += 1

								#Recalculating speed and saving it in an ordered dict, also assigning 
								#values to agent's current variables
								a.current_edge = edge_name
								a.current_edge_num = edge_num
								a.current_path = path_num
								a.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, agent = a)
								a.speed_by_time[t] = int(a.speed)
								a.edges_passed[path_num].append(edge_name)

								#Checking prints
								#print('----------COMING----------')
								#print('agent id', a.id_num)
								#print('number of cars before', num_of_cars_before_agent_on_an_edge)
								#print('time',t)
								#print('speed', a.speed)
								#print('edge', a.current_edge)

								#Agent estimates his time of arrival to the end of the edge taking into account 
								#current speed on edge for him
								self.calculate_real_edge_time_for_edge(agent = a, 
																		path_num = path_num, 
																		edge_num = edge_num)

							#Postponing departure times for next trips if they are earlier then expected
							#time of current path finish
							if path_num <= len(a.paths)-2 and a.real_edge_times[path_num][edge_num][-1] != None:
								for i in range(1,len(a.paths)-path_num):
									if a.real_edge_times[path_num][edge_num][-1] >= a.times_of_departures[path_num+i]:
										a.times_of_departures[path_num+i] = a.real_edge_times[path_num][edge_num][-1] + 0.01*i
										a.times_of_departures[path_num+i] = \
										float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) \
											+ "f}").format(round(a.times_of_departures[path_num+i], digits_to_save)))

			#Delete variables from memory
			#del a
			#del path_num

			if arch_time == int((self.max_time/self.agents[0].time_stamp)):
				self.check_agents_prolong_simulation()
				#print(self.max_time)
		
		#Sorting speed_by_time dictionary for all agents by time
		self.sort_speed_by_time_for_all()

		#Test for consistency in simulation 
		#(number of cars came to edge must be equal to number of cars left the edge)
		#print("Equality:", edge_name_came == edge_name_left)
		if (edge_name_came == edge_name_left) == False:
			print('Equality violated')
			print('Came', edge_name_came)
			print('Left', edge_name_left)


	#Fucntion that copies agents and saves statistics about them
	def copy_agents(self):
		copied_agents = []
		for a in self.agents:
			copied_agents.append(deepcopy(a))
		return copied_agents


	#Function that restores every agent's parameters to initial 
	#ones (uses function agent.restore()
	#from class Agent)
	def restore_all_agents(self):
		for a in self.agents:
			a.restore()


	#Function that restores every agent's parameters to initial ones, 
	#also restoring current_iteration parameter
	def restore_all_agents_finally(self):
		for a in self.agents:
			a.restore(finish = True)


	#Main function that repeatedly makes iterations of the simulation to 
	#reach the common-sence equilibrium with shortest-distance paths
	def simulation_optimized(self, max_iters = 100):
		arch_start = time.time()
		self.number_of_iterations = max_iters
		for iteration in range(self.number_of_iterations):
			start = time.time()
			self.common_sense[iteration] = self.copy_agents()
			self.simulation_iteration()
			self.common_sense[iteration] = self.copy_agents()
			self.restore_all_agents()
			self.paths_changed_from_last_iter()
			if self.iteration >= 1 and self.paths_changed[self.iteration] == 0:
				self.iteration += 1
				self.number_of_iterations = self.iteration
				print('Agents paths converged to user-equilibrium')
				end = time.time()
				print('End of iteration', iteration, '. Time elapsed:', round((end - start), 2), 'sec')
				break
			self.iteration += 1
			end = time.time()
			print('End of iteration', iteration, '. Time elapsed:', round((end - start), 2), 'sec')
		self.statistics[0] = deepcopy(self.common_sense[len(self.common_sense)-1])
		self.restore_all_agents_finally()
		arch_end = time.time()
		print('End of simulation', '. Time elapsed:', round((arch_end - arch_start), 2), 'sec')


	#Function that counts the number of paths that were changed from previous
	#to current iteration
	def paths_changed_from_last_iter(self):
		if self.iteration >= 1:
			self.paths_changed[self.iteration] = 0
			for iteration in range(self.iteration, len(self.common_sense)):
				for a in self.common_sense[iteration]:
					for p in range(0,len(a.paths)):
						if a.paths[p] != self.common_sense[iteration-1][a.id_num].paths[p]:
							self.paths_changed[iteration] += 1
		else:
			pass



	#Function that returns data from specified iteration's congestion on each
	#edge at each time period specified by time_stamp in Agent 
	#class initialization
	def get_congestion_data_for_edges(self, iteration):
		congestion = [dict((edge,0) for edge in self.graph.edges()) for t in \
		range(0,int(self.max_time/self.agents[0].time_stamp))]

		for t in range(0, int(self.max_time/self.agents[0].time_stamp)):
			idx = t
			t = float(t/100)

			for a in self.statistics[iteration]:
				for p in range(len(a.paths)):
					for edge_num in range(len(a.paths[p])-1):
						if t >= a.real_edge_times[p][edge_num][0] and \
						t < a.real_edge_times[p][edge_num][1]:
							edge_name = a.edges_passed[p][edge_num]
							congestion[idx][edge_name] += 1

		return congestion


	#Function that searches across all agents and identifies ones with 
	#incomplete paths. After such identification function assigns new
	#maximim time for the simulation in order for all drivers to finish their trips
	def check_agents_prolong_simulation(self):
		for a in self.agents:
			for path_num in range(len(a.paths)):
				if a.paths[path_num] == None:
					if self.max_time < a.times_of_departures[path_num]:
						self.max_time = a.times_of_departures[path_num]
						#print('Changed max_time', self.max_time)
					else:
						pass
				else:
					for edge_num in range(len(a.paths[path_num]) - 1):
						if a.real_edge_times[path_num][edge_num][-1] != None \
						and a.real_edge_times[path_num][edge_num][-1] > self.max_time:
							self.max_time = a.real_edge_times[path_num][edge_num][-1]
							break




class simulation_navigator(simulation_baseline):
	def __init__(self, graph, agents, max_time, common_sense):
		simulation_baseline.__init__(self, graph, agents, 1, max_time)

		#Parameters initialization
		self.graph = graph
		self.agents = agents
		self.max_time = max_time
		self.statistics = dict()
		self.iteration = 0
		self.number_of_iterations = 1

		self.common_sense = common_sense
		del common_sense[len(common_sense)-1]
		self.cars_ahead = dict((edge, [[60, 0]]) for edge in self.graph.edges())

		prop_proxy = 0
		for agent in self.agents:
			if agent.navigator_user == True:
				prop_proxy += 1

		self.proportion_users = prop_proxy / len(self.agents)
		del prop_proxy

	#Function that updates weights of the edges for navigator
	def update_weights(self, agents_on_edges, current_time):
		for edge_key, edge_list in agents_on_edges.items():
			distance_preserver = []
			update_list = []
			num_navi_agents = 0
			for agent in edge_list:
				if agent.navigator_user == True:
					num_navi_agents += 1
			for agent in edge_list:
				if agent.navigator_user == True:
				
					passed_distance = self.calculate_passed_distance(agent = agent,
																	current_time = current_time)
					distance_to_pass = self.graph[edge_key[0]][edge_key[1]]['weight']\
					- passed_distance

					distance_preserver.append(distance_to_pass)
					if len(distance_preserver) == 1:
						update_val = (distance_preserver[-1] / agent.speed) * 60
					else:
						update_val = ((distance_preserver[-1] - \
									distance_preserver[-2]) / agent.speed) * 60
					update_list.append(update_val)
					if len(update_list) == num_navi_agents:
						update_list.append((passed_distance / agent.speed) * 60)
			if update_list != []:
				self.graph[edge_key[0]][edge_key[1]]['navigator_weight'] = \
				sum(update_list)
			
			del update_list
			del distance_preserver

		
	#Function that calculates expected edge times for path->edge
	def calculate_expected_edge_time_for_edge(self, agent, path_num, edge_num):
		digits_to_save = len(str(agent.time_stamp)) - 2
		if edge_num == 0:
			#Expected edge time is calculated as the edge subjective (for agent) weight 
			#divided by 60 kmh (default speed). Expected edge time is a time in which 
			#agent expects to finish traversing the edge
			#Expected times are different for those who use navigator and those who don't
			if agent.navigator_user == True:
				time = agent.times_of_departures[path_num]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (self.graph[edge[0]][edge[1]]['navigator_weight'] / agent.speed))))
			else:
				time = agent.times_of_departures[path_num]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (calculate_subjective_weight_for_edge_time_overall(graph = self.graph,
																			agent = agent, 
																			time = time,
																			edge = edge,
																			statistics = self.common_sense) \
				/ agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			agent.real_edge_times[path_num].append(list((time, None)))
		
		else:
			if agent.navigator_user == True:
				time = agent.expected_edge_times[path_num][edge_num-1][1]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (self.graph[edge[0]][edge[1]]['navigator_weight'] / agent.speed))))
			else:
				time = agent.expected_edge_times[path_num][edge_num-1][1]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (calculate_subjective_weight_for_edge_time_overall(graph = self.graph,
																			agent = agent, 
																			time = time,
																			edge = edge,
																			statistics = self.common_sense) \
				/ agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			agent.real_edge_times[path_num].append(list((None, None)))
		#Everything is rounded to maintain discrete time consistency
		for fl in range(2):
			agent.expected_edge_times[path_num][edge_num][fl] = \
			float(("{0:." + str(digits_to_save) + "f}").format(round(agent.expected_edge_times[path_num][edge_num][fl], digits_to_save)))
		agent.real_edge_times[0][0][0] = \
		float(("{0:." + str(digits_to_save) + "f}").format(round(agent.real_edge_times[0][0][0], digits_to_save)))


	#LAST CHANGES:
	#2) cntr instead of index(agent)
	#3) Direct path adressing for agent's path instead of agent.class_method()
	#Function which makes 1 iteration of navigator_simulation
	def simulation_iteration_navigator(self):

		digits_to_save = len(str(self.agents[0].time_stamp)) - 2
		update_list = dict((e,[]) for e in self.graph.edges())

		#Priorty qeues for every edge and counters for each edge to check whether every agent who
		#came to edge also left it
		agents_on_edges = dict((e,[]) for e in self.graph.edges())
		edge_name_came = dict((e, 0) for e in self.graph.edges())
		edge_name_left = dict((e, 0) for e in self.graph.edges())

		#Main Loop that makes each iteration of the loop tick with the time interval specified by time_stamp
		#for i in range(0,int((self.max_time/self.agents[0].time_stamp))):
		#	t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
		#		"f}").format(round((0 + i*self.agents[0].time_stamp), digits_to_save)))
		arch_time = -1
		while arch_time != int((self.max_time/self.agents[0].time_stamp)):
			arch_time += 1
			t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
				"f}").format(round((0 + arch_time*self.agents[0].time_stamp), digits_to_save)))

			#Agent leaves the edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num]) - 1):

							#Agent leaves the edge
							if a.real_edge_times[path_num][edge_num][1] == t:

								#Find the edge from which agent leaves and decrease the number of cars on it by one
								#Also, dequeue agent from this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])
								index = agents_on_edges[edge_name].index(a)
								agents_on_edges[edge_name].pop(index)
								#if index != 0:
									#print('Time', t, 'Agent', a.id_num, 'IDX', index)
									#agents_on_edges[edge_name].pop(index)
								#if index == 0:
									#agents_on_edges[edge_name].pop(0)

								#Check
								edge_name_left[edge_name] += 1

								#Checking prints
								#print('----------LEAVING----------')
								#print('agent id', a.id_num)
								#print('number_of_cars_after', len(agents_on_edges[edge_name]))
								#print('time',t)
								#print('edge',edge_name)

								#Cleans current edge variables and transfers time of edge-finish to next edge start time
								#Or in case this is the end the path for agent increases his current path number by one
								a.current_edge = (0,0)
								a.current_edge_num = []
								if edge_num <= (len(a.real_edge_times[path_num])-2):
									a.real_edge_times[path_num][edge_num+1][0] = a.real_edge_times[path_num][edge_num][1]
								if edge_num == len(a.real_edge_times[path_num])-1:
									a.current_path += 1
								
								#Recalculating real travel times for every agent on an edge 
								#(who is not leacing at the current time) from which another agent left
								#Their speed must change due to decrese in number of cars ahead of them
								cntr = -1
								for agent in agents_on_edges[edge_name]:
									cntr += 1
									if agent.current_edge == edge_name and \
									(agent.real_edge_times[agent.current_path][agent.current_edge_num][1] != t):

										#Searching for the position of the agent on an edge and finding number of cars ahead
										num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name][:int(cntr)])

										#Recalculating speed and saving it in an ordered dict
										agent.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, agent = agent)
										agent.speed_by_time[t] = int(agent.speed)
										self.cars_ahead[edge_name].append([agent.speed, num_of_cars_before_agent_on_an_edge])

										#Calculating the distance on edge which is already passed by an agent, distance to pass,
										#and finally updating the real travel time on the edge
										passed_distance = self.calculate_passed_distance(agent = agent, 
																							current_time = t)

										weight_of_the_edge = self.graph[edge_name[0]][edge_name[1]]['weight']
										distance_to_pass = round(weight_of_the_edge - passed_distance, digits_to_save)

										#Checking prints
										#print('----------------')
										#print('Time', t)
										#print('agent id', agent.id_num)
										#print('Path', agent.current_path)
										#print("Current edge", agent.current_edge)
										#print('Time of arrival before change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Passed Distance', passed_distance)

										time_to_pass = float(("{0:." + str(digits_to_save) + "f}").\
											format(round(distance_to_pass / agent.speed, digits_to_save)))
										#Small fix to secure problems with rounding
										if time_to_pass == 0.0:
											time_to_pass = 0.01
										agent.real_edge_times[agent.current_path][agent.current_edge_num][1] = \
										float(("{0:." + str(digits_to_save) + "f}").format(round(t + time_to_pass, digits_to_save)))
										
										#Checking prints continued
										#print('Time of arrival after change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Speed after', agent.speed)
										#print('NUM_BOYS', num_of_cars_before_agent_on_an_edge)
								#Break edge_num loop
								break
			#Delete variables from memory
			#del a
			#del path_num

			#Update navigator weights
			self.update_weights(agents_on_edges = agents_on_edges,
								current_time = t)

			#Agent finds path and expects to traverse it for some time Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					
					#If it is time to go - agent finds the path from source to target
					if a.times_of_departures[path_num] == t \
					and a.paths[path_num] == None and a.current_edge == (0,0):
						if a.navigator_user == True:
							a.paths[path_num] = nx.dijkstra_path(G = self.graph, 
												source = a.start_nodes[path_num], 
												target = a.finish_nodes[path_num], 
												weight = 'navigator_weight')
						else:
							a.paths[path_num] = \
							dijkstra_with_time_dependent_weights(	graph = self.graph, 
																	source = a.start_nodes[path_num],
																	target = a.finish_nodes[path_num], 
																	agent = a, 
																	time = t, 
																	statistics = self.common_sense, 
																	length = False	)
						
						#To maintain expectation correctness we need to increase agent's speed to 60 kmh
						a.speed = 60
						#Then he finds the expected time he will be on each edge, also, expected travel time
						for edge_num in range(len(a.paths[path_num])-1):
							self.calculate_expected_edge_time_for_edge(agent = a,
																		path_num = path_num, 
																		edge_num = edge_num)


						if path_num <= len(a.paths)-2:
							for i in range(1,len(a.paths)-path_num):
								if a.expected_edge_times[path_num][-1][-1] >= a.times_of_departures[path_num+i]:
									a.times_of_departures[path_num+i] = float(a.expected_edge_times[path_num][-1][-1]) + 0.01*i
									a.times_of_departures[path_num+i] = \
									float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + "f}")\
										.format(round(a.times_of_departures[path_num+i], digits_to_save)))
			
			#Delete variables from memory
			#del a
			#del path_num

			#Agent comes to edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num])-1):
							
							#Agent comes to edge
							if a.real_edge_times[path_num][edge_num][0] == t and \
							a.current_edge == (0,0) and a.current_path == path_num:

								#Find the edge on which agent comes and increase the number of cars on it by one
								#Also, enqueue the agent on this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])
								num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name])
								agents_on_edges[edge_name].append(a)

								#Increase the daily number of cars came to the edge
								edge_name_came[edge_name] += 1

								#Recalculating speed and saving it in an ordered dict, also assigning 
								#values to agent's current variables
								a.current_edge = edge_name
								a.current_edge_num = edge_num
								a.current_path = path_num
								a.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, 
																agent = a)
								a.speed_by_time[t] = int(a.speed)
								a.edges_passed[path_num].append(edge_name)
								self.cars_ahead[edge_name].append([a.speed, num_of_cars_before_agent_on_an_edge])

								#Checking prints
								#print('----------COMING----------')
								#print('agent id', a.id_num)
								#print('number of cars before', num_of_cars_before_agent_on_an_edge)
								#print('time',t)
								#print('speed', a.speed)
								#print('edge', a.current_edge)

								#Agent estimates his time of arrival to the end of the edge taking into account 
								#current speed on edge for him
								self.calculate_real_edge_time_for_edge(agent = a, 
																		path_num = path_num, 
																		edge_num = edge_num)

							#Postponing departure times for next trips if they
							#are earlier then expected finish time of current path
							if path_num <= len(a.paths)-2 and a.real_edge_times[path_num][edge_num][-1] != None:
								for i in range(1,len(a.paths)-path_num):
									if a.real_edge_times[path_num][edge_num][-1] >= a.times_of_departures[path_num+i]:
										a.times_of_departures[path_num+i] = a.real_edge_times[path_num][edge_num][-1] + 0.01 * i
										a.times_of_departures[path_num+i] = \
										float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) \
											+ "f}").format(round(a.times_of_departures[path_num+i], digits_to_save)))

			#Delete variables from memory
			#del a
			#del path_num
			if arch_time == int((self.max_time/self.agents[0].time_stamp)):
				self.check_agents_prolong_simulation()
				#print(self.max_time)

		#Sorting speed_by_time dictionary for all agents by time
		self.sort_speed_by_time_for_all()

		#Test for consistency in simulation 
		#(number of cars came to edge must be equal to number of cars left the edge)
		#print("Equality:", edge_name_came == edge_name_left)
		if (edge_name_came == edge_name_left) == False:
			print('Equality violated')
			print('Came', edge_name_came)
			print('Left', edge_name_left)


	#Function that simulates navigator-driven traffic
	def simulation_optimized(self, max_iters = 20):
		#If proportion of navigator users is 1, then there is no need
		#for evolutionary optimization as equilibrium is deterministic
		#and will not change
		arch_start = time.time()
		#Preparation activities
		for a in self.agents:
			a.current_iter = len(self.common_sense)
		if self.proportion_users == 1 or self.proportion_users == 0:
			self.common_sense[len(self.common_sense)] = self.copy_agents()
			self.simulation_iteration_navigator()
			self.common_sense[len(self.common_sense) - 1] = self.copy_agents()
			self.statistics[0] = self.copy_agents()
			self.restore_all_agents()
			self.restore_all_agents_finally()
		#If proportion of navigator users is neither 1 nor 0
		#the 0-th iteration traffic situation is not an equilibrium
		# and thus needs to be evulutionary optimized in order to obtain
		#real equolibrium where not all drivers use navigator
		else:
			self.number_of_iterations = max_iters
			self.iteration = len(self.common_sense)
			for iteration in range(self.number_of_iterations):
				start = time.time()
				self.common_sense[self.iteration] = self.copy_agents()
				self.simulation_iteration_navigator()
				self.common_sense[self.iteration] = self.copy_agents()
				if iteration == self.number_of_iterations - 1:
					#self.statistics[0] = self.copy_agents()
					self.statistics[0] = deepcopy(self.common_sense[len(self.common_sense) - 1])
				self.restore_all_agents()
				self.paths_changed_from_last_iter()
				if self.paths_changed[self.iteration] == 0:
					self.iteration += 1
					self.number_of_iterations = self.iteration
					print('Agents paths converged to user-equilibrium')
					end = time.time()
					print('End of iteration', iteration, \
						'. Time elapsed:', round((end - start), 2), 'sec')
					#self.statistics[0] = self.copy_agents()
					self.statistics[0] = deepcopy(self.common_sense[len(self.common_sense) - 1])
					break
				self.iteration += 1
				end = time.time()
				print('End of iteration', iteration, \
					'. Time elapsed:', round((end - start), 2), 'sec')
			self.restore_all_agents_finally()
		arch_end = time.time()
		print('End of simulation', '. Time elapsed:', \
			round((arch_end - arch_start), 2), 'sec')


	#Function that counts the number of paths that were changed from previous
	#to current iteration
	def paths_changed_from_last_iter(self):
		if len(self.common_sense) - self.iteration >= 1:
			self.paths_changed[self.iteration] = 0
			for iteration in range(self.iteration, len(self.common_sense)):
				for a in self.common_sense[iteration]:
					for p in range(0,len(a.paths)):
						if a.paths[p] != self.common_sense[iteration-1][a.id_num].paths[p]:
							self.paths_changed[iteration] += 1
		else:
			pass


	#Function that returns data on first iteration's congestion on each 
	#edge at each time period specified by time_stamp in Agent 
	#class initialization
	def get_congestion_data_for_edges(self, iteration = 0):
		congestion = [dict((edge,0) for edge in self.graph.edges()) for t in \
		range(0,int(self.max_time/self.agents[0].time_stamp))]

		for t in range(0, int(self.max_time/self.agents[0].time_stamp)):
			idx = t
			t = float(t/100)

			for a in self.statistics[iteration]:
				for p in range(len(a.paths)):
					for edge_num in range(len(a.paths[p])-1):
						if t >= a.real_edge_times[p][edge_num][0] and \
						t < a.real_edge_times[p][edge_num][1]:
							edge_name = a.edges_passed[p][edge_num]
							congestion[idx][edge_name] += 1

		return congestion


	#Function that returns data on first iteration's congestion gathered 
	#only by those agents who use navigator on each edge at each time 
	#period specified by time_stamp in Agent class initialization
	def get_congestion_data_for_edges_observed(self):
		congestion = [dict((edge,0) for edge in self.graph.edges()) for t in \
		range(0,int(self.max_time/self.agents[0].time_stamp))]

		counter = 0
		for a in self.agents:
			if a.navigator_user == True:
				counter +=1 
		del a
		proportion_users = counter/len(self.agents)
		
		statistics_available = deepcopy(self.statistics[0])
		for agent in statistics_available:
			if agent.navigator_user == False:
				statistics_available.remove(agent)
		del agent

		for t in range(0, int(self.max_time/self.agents[0].time_stamp)):
			idx = t
			t = float(t/100)

			for a in statistics_available:
				for p in range(len(a.paths)):
					for edge_num in range(len(a.paths[p])-1):
						if t >= a.real_edge_times[p][edge_num][0] and \
						t < a.real_edge_times[p][edge_num][1]:
							edge_name = a.edges_passed[p][edge_num]
							congestion[idx][edge_name] += round(1 / proportion_users)

		#for time_dict in congestion:
		#	for edge_key, edge_val in time_dict.items():
		#		edge_val = round(edge_val * (1/proportion_users))

		return congestion




class simulation_PTA(simulation_baseline):
	def __init__(self, graph, agents, max_time, historical_data, common_sense, cars_ahead):
		simulation_baseline.__init__(self, graph, agents, 1, max_time)

		self.graph = graph
		self.agents = agents
		self.max_time = max_time
		self.statistics = dict()
		self.common_sense = common_sense
		del common_sense[len(common_sense)-1]

		self.historical_data = historical_data

		self.cars_ahead = cars_ahead
		self.min_speed = dict()
		for edge in self.cars_ahead.keys():
			self.min_speed[edge] = min(np.array(self.cars_ahead[edge])[:,0])
		self.max_speed = 60

		self.trips_data = dict((e, dict((t, [0, []]) \
			for t in range(0, int(max_time / 0.01) + 1))) \
			for e in self.graph.edges())
		
		prop_proxy = 0
		for agent in self.agents:
			if agent.PTA_user == True:
				prop_proxy += 1

		self.proportion_users = prop_proxy / len(self.agents)
		del prop_proxy


	#Function that transforms initial historical data about agents to the dict
	#which has number of cars on every edge at every time-momemt
	def transform_historical_data(self):
		max_time_val = []
		for a in self.historical_data:
			for p in range(len(a.paths)):
				finish_time = a.real_edge_times[p][-1][-1]
				max_time_val.append(finish_time)
		max_time_val = max(max_time_val)
		if max_time_val > self.max_time:
			self.max_time = max_time_val
			self.trips_data = dict((e, dict((t, [0, []]) \
			for t in range(0, int(self.max_time / 0.01) + 1))) \
			for e in self.graph.edges())

		self.cong_hist_data = dict((e, dict((t, [0, []]) \
			for t in range(0, int(self.max_time / 0.01) + 1))) \
			for e in self.graph.edges())
		
		for a in self.historical_data:
			for p in range(len(a.paths)):
				for edge_num in range(len(a.paths[p])-1):
					start_time = int(round(a.real_edge_times[p][edge_num][0] / 0.01))
					finish_time = int(round(a.real_edge_times[p][edge_num][1] / 0.01))
					edge = a.edges_passed[p][edge_num]
					for time in range(start_time, finish_time + 1):
						self.cong_hist_data[edge][time][0] += 1
						#If agent leaves the edge at time = t than it is indicated
						if time == finish_time:
							self.cong_hist_data[edge][time][1].append('l')
			

	#Function that calculates the subjective weight for given edge and time of
	#arrival using the data available for PTA server
	def calculate_weight_for_edge_time(self, arrival_time, edge):
		min_speed = self.min_speed[edge]
		max_speed = 60
		time_stamp = self.agents[0].time_stamp
		weight_obj = 0
		weight_subj = 0
		t = 0
		arr_time = int(round(arrival_time / time_stamp))
		position = round((self.trips_data[edge][arr_time][0] * self.proportion_users) \
		+ (self.cong_hist_data[edge][arr_time][0] * (1 - self.proportion_users)))
		
		while weight_obj < self.graph[edge[0]][edge[1]]['weight']:
			t = round(t, len(str(self.agents[0].time_stamp)) - 2)
			time = int(round((round(arrival_time + t, \
				len(str(self.agents[0].time_stamp)) - 2) / time_stamp)))
			#num_cars_ahead = round((self.trips_data[edge][time][0] * self.proportion_users) \
			#+ (self.cong_hist_data[edge][time][0] * (1 - self.proportion_users)))
			#speed = self.reg2[edge].predict(num_cars_ahead)
			left = round(len(self.trips_data[edge][time][1]) * self.proportion_users \
			+ len(self.cong_hist_data[edge][time][1]) * (1 - self.proportion_users))
			position -= left
			if position <= 0:
				position = 0
			#print('Pos', position, 'Left', left)

			speed = self.reg2[edge].predict(position)
			if speed <= min_speed:
				speed = min_speed
			if speed >= max_speed:
				speed = max_speed
			
			weight_obj += speed * time_stamp
			weight_subj += 60 * time_stamp
			t += 0.01
			if t >= 28:
				break
			#print('Time', time, 'WO', weight_obj, 'WS', weight_subj, 'speed', speed)
			#print('Cars ahead', num_cars_ahead)

		if weight_obj >= self.graph[edge[0]][edge[1]]['weight']:
			diff = weight_obj - self.graph[edge[0]][edge[1]]['weight']
			time_diff = diff / speed
			weight_obj -= diff
			weight_subj -= time_diff * 60
			return float(weight_subj)

		return float(weight_subj)


	#Function that calculates the subjective weight for given edge and time of
	#arrival using the data available for PTA server
	def calculate_weight_for_edge_time_diff(self, arrival_time, edge):
		min_speed = self.min_speed[edge]
		max_speed = 60
		time_stamp = self.agents[0].time_stamp
		weight_obj = 0
		weight_subj = 0
		t = int(round((arrival_time / time_stamp)))
		time_saved = int(round(arrival_time / time_stamp))

		try:
			position = round((self.trips_data[edge][t][0] * self.proportion_users) \
			+ (self.cong_hist_data[edge][t][0] * (1 - self.proportion_users)))
			left = round(len(self.trips_data[edge][t][1]) * self.proportion_users \
				+ len(self.cong_hist_data[edge][t][1]) * (1 - self.proportion_users))
		except KeyError:
			for time_proxy in range(len(self.trips_data[edge]), t + 1):
				self.cong_hist_data[edge][time_proxy] = [0, []]
				self.cong_hist_data[edge][time_proxy] = [0, []]
			position = round((self.trips_data[edge][t][0] * self.proportion_users) \
			+ (self.cong_hist_data[edge][t][0] * (1 - self.proportion_users)))
			left = round(len(self.trips_data[edge][t][1]) * self.proportion_users \
				+ len(self.cong_hist_data[edge][t][1]) * (1 - self.proportion_users))

		position -= left
		if position <= 0:
			position = 0
		position_saved = position

		speed = self.reg2[edge].predict(position_saved)
		if speed <= self.min_speed[edge]:
			speed = self.min_speed[edge]
		if speed >= self.max_speed:
			speed = self.max_speed

		while weight_obj < self.graph[edge[0]][edge[1]]['weight']:
			
			while position == position_saved:
				t += 1
				if t >= int(round(self.max_time / 0.01)):
					break
				left = round(len(self.trips_data[edge][t][1]) * self.proportion_users \
					+ len(self.cong_hist_data[edge][t][1]) * (1 - self.proportion_users))
				position -= left
				if position <= 0:
					position = 0
			
			time_passed = t - time_saved
			time_saved = t

			weight_obj += round(float(speed * (time_passed * time_stamp)), \
				len(str(time_stamp)) - 2)
			weight_subj += round(60 * (time_passed * time_stamp), \
				len(str(time_stamp)) - 2)

			speed = self.reg2[edge].predict(position)
			if speed <= self.min_speed[edge]:
				speed = self.min_speed[edge]
			if speed >= self.max_speed:
				speed = self.max_speed

			position_saved = position


		if weight_obj >= self.graph[edge[0]][edge[1]]['weight']:
			diff = weight_obj - self.graph[edge[0]][edge[1]]['weight']
			time_diff = diff / speed
			weight_obj -= diff
			weight_subj -= time_diff * 60
			return round(float(weight_subj), len(str(time_stamp)) - 2)
		if weight_obj <= self.graph[edge[0]][edge[1]]['weight']:
			weight_obj = self.graph[edge[0]][edge[1]]['weight']
			weight_subj = self.graph[edge[0]][edge[1]]['weight']
			return round(float(weight_subj), len(str(time_stamp)) - 2)

		return round(float(weight_subj), len(str(time_stamp)) - 2)


	#Expected edge times
	def calculate_expected_edge_time_for_edge(self, agent, path_num, edge_num):
		digits_to_save = len(str(agent.time_stamp)) - 2
		if edge_num == 0:
			#Expected edge time is calculated as the edge subjective (for agent) 
			#weight divided by 60 kmh (default speed)
			#Expected edge time is a time in which agent expects to finish traversing the edge
			#Expected times are different for those who use navigator and those who don't
			if agent.PTA_user == True:
				time = agent.times_of_departures[path_num]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				time_range = \
				self.calculate_weight_for_edge_time_diff(arrival_time = time, 
															edge = edge) / agent.speed
				time_range = float(("{0:." + str(digits_to_save) \
				+ "f}").format(round(time_range, digits_to_save)))

				agent.expected_edge_times[path_num].append(list((time, time + time_range)))
				#Update trips data on the PTA server
				for t in range(int(round(time / agent.time_stamp)), \
					int(round((time + time_range) / agent.time_stamp)) + 1):
					#if t >= (self.max_time / 0.01):
					#	self.trips_data[edge][t] = [0,[]]
					#	self.cong_hist_data[edge][t] = [0, []]
					self.trips_data[edge][t][0] += 1
					#If agent leaves the edge at time = t than it is indicated
					if t == int(round((time + time_range) / agent.time_stamp)):
						self.trips_data[edge][t][1].append('l')
			else:
				time = agent.times_of_departures[path_num]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (calculate_subjective_weight_for_edge_time_overall(graph = self.graph,
																		agent = agent,
																		time = time,
																		edge = edge,
																		statistics = self.common_sense) \
				/ agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			agent.real_edge_times[path_num].append(list((time, None)))
		
		else:
			if agent.PTA_user == True:
				time = agent.expected_edge_times[path_num][edge_num-1][1]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				time_range = self.calculate_weight_for_edge_time_diff(arrival_time = time, 
																		edge = edge) / agent.speed
				time_range = float(("{0:." + str(digits_to_save) \
				+ "f}").format(round(time_range, digits_to_save)))
				agent.expected_edge_times[path_num].append(list((time, time + time_range)))
				#Update trips data on the PTA server
				for t in range(int(round(time / agent.time_stamp)), \
					int(round((time + time_range) / agent.time_stamp)) + 1):
					#if t >= (self.max_time/0.01):
					#	self.trips_data[edge][t] = [0,[]]
					#	self.cong_hist_data[edge][t] = [0, []]
					self.trips_data[edge][t][0] += 1
					#If agent leaves the edge at time = t than it is indicated
					if t == int(round((time + time_range) / agent.time_stamp)):
						self.trips_data[edge][t][1].append('l')
			else:
				time = agent.expected_edge_times[path_num][edge_num-1][1]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (calculate_subjective_weight_for_edge_time_overall(graph = self.graph, \
																		agent = agent, 
																		time = time,
																		edge = edge,
																		statistics = self.common_sense) \
				/ agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			agent.real_edge_times[path_num].append(list((None, None)))
		#Everything is rounded to maintain discrete time consistency
		for fl in range(2):
			agent.expected_edge_times[path_num][edge_num][fl] = \
			float(("{0:." + str(digits_to_save) + \
				"f}").format(round(agent.expected_edge_times[path_num][edge_num][fl], digits_to_save)))
		agent.real_edge_times[0][0][0] = \
		float(("{0:." + str(digits_to_save) + \
			"f}").format(round(agent.real_edge_times[0][0][0], digits_to_save)))


	#Function that connects number of cars to speed (regression\AdaBoost)
	def get_relationship_cars_ahead_speed(self):
		data_sets = deepcopy(self.cars_ahead)
		self.reg = dict((e,0) for e in self.graph.edges())
		self.reg2 = dict((e,0) for e in self.graph.edges())
		for edge, data in data_sets.items():
			if data_sets[edge] == []:
				data_sets[edge].append([60, 0])
			data_sets[edge] = np.array(data_sets[edge])
			y = data_sets[edge][:,0].reshape(len(data_sets[edge]), 1).ravel()
			X = data_sets[edge][:,1].reshape(len(data_sets[edge][:,1]), 1)
			# AdaBoost to predict speed depending on number of cars ahead
			self.reg2[edge] = AdaBoostRegressor(n_estimators = 50, 
												learning_rate = 1.0)
			self.reg2[edge].fit(X, y)


	#Classical Dijkstra algorithm which uses time-dependent weights
	#supplied by PTA server
	def dijkstra_PTA(self, source, target, time, length = False):
	
		digits_to_save = len(str(self.agents[0].time_stamp)) - 2
		
		adj_list = self.graph.adjacency_list()

		dist = [float('inf')]*len(adj_list)
		prev = [None]*len(adj_list)
		dist[source-1] = 0
		prev[source-1] = source - 1

		dist_proxy = [(0, source - 1)]
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
				time = float(("{0:." + str(digits_to_save) + "f}").format \
					(round(init_time + (dist[min_node] / 60), digits_to_save)))
				#print(time)
			
			for vertex in adj_list[min_node]:
				vertex = vertex-1
				cost = self.calculate_weight_for_edge_time_diff(arrival_time = time, 
															edge = (min_node+1, vertex+1))
					
				if dist[vertex] > dist[min_node] + cost:
					proxy_var = dist[vertex]
					dist[vertex] = dist[min_node] + cost
					heappush(dist_proxy, (dist[vertex], vertex))
					prev[vertex] = min_node

	#LAST CHANGES:
	#2) cntr instead of index(agent)
	#3) Direct path adressing for agent's path instead of agent.class_method()
	#Simulation iteration
	def simulation_iteration_PTA(self):

		digits_to_save = len(str(self.agents[0].time_stamp)) - 2

		#Priorty qeues for every edge and counters for each edge to check whether every agent who
		#came to edge also left it
		agents_on_edges = dict((e,[]) for e in self.graph.edges())
		edge_name_came = dict((e, 0) for e in self.graph.edges())
		edge_name_left = dict((e, 0) for e in self.graph.edges())

		#Main Loop that makes each iteration of the loop tick with the time interval specified by time_stamp
		#for i in range(0,int((self.max_time/self.agents[0].time_stamp))):
		#	t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
		#		"f}").format(round((0 + i*self.agents[0].time_stamp), digits_to_save)))
		arch_time = -1
		while arch_time != int((self.max_time/self.agents[0].time_stamp)):
			arch_time += 1
			t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
				"f}").format(round((0 + arch_time*self.agents[0].time_stamp), digits_to_save)))

			#Agent leaves the edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num])-1):

							#Agent leaves the edge
							if a.real_edge_times[path_num][edge_num][1] == t:

								#Find the edge from which agent leaves and decrease the number of cars on it by one
								#Also, dequeue agent from this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])
								index = agents_on_edges[edge_name].index(a)
								agents_on_edges[edge_name].pop(index)
								#if index != 0:
									#print('Time', t, 'Agent', a.id_num, 'IDX', index)
								#	agents_on_edges[edge_name].pop(index)
								#if index == 0:
								#	agents_on_edges[edge_name].pop(0)

								#Check
								edge_name_left[edge_name] += 1

								#Checking prints
								#print('----------LEAVING----------')
								#print('agent id', a.id_num)
								#print('number_of_cars_after', len(agents_on_edges[edge_name]))
								#print('time',t)
								#print('edge',edge_name)

								#Cleans current edge variables and transfers time of edge-finish to next edge start time
								#Or in case this is the end the path for agent increases his current path number by one
								a.current_edge = (0,0)
								a.current_edge_num = []
								if edge_num <= (len(a.real_edge_times[path_num])-2):
									a.real_edge_times[path_num][edge_num+1][0] = a.real_edge_times[path_num][edge_num][1]
								if edge_num == len(a.real_edge_times[path_num])-1:
									a.current_path += 1
								
								#Recalculating real travel times for every agent on an edge 
								#(who is not leacing at the current time) from which another agent left
								#Their speed must change due to decrese in number of cars ahead of them
								cntr = -1
								for agent in agents_on_edges[edge_name]:
									cntr += 1
									if agent.current_edge == edge_name and \
									(agent.real_edge_times[agent.current_path][agent.current_edge_num][1] != t):

										#Searching for the position of the agent on an edge and finding number of cars ahead
										num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name][:int(cntr)])

										#Recalculating speed and saving it in an ordered dict
										agent.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, agent = agent)
										agent.speed_by_time[t] = int(agent.speed)

										#Calculating the distance on edge which is already passed by an agent, distance to pass,
										#and finally updating the real travel time on the edge
										passed_distance = self.calculate_passed_distance(agent = agent, 
																							current_time = t)

										weight_of_the_edge = self.graph[edge_name[0]][edge_name[1]]['weight']
										distance_to_pass = weight_of_the_edge - passed_distance

										#Checking prints
										#print('----------------')
										#print('Time', t)
										#print('agent id', agent.id_num)
										#print('Path', agent.current_path)
										#print("Current edge", agent.current_edge)
										#print('Time of arrival before change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Passed Distance', passed_distance)

										time_to_pass = float(("{0:." + str(digits_to_save) \
											+ "f}").format(distance_to_pass / agent.speed))
										#Small fix to secure problems with rounding
										if time_to_pass == 0.0:
											time_to_pass = 0.01
										agent.real_edge_times[agent.current_path][agent.current_edge_num][1] = \
										float(("{0:." + str(digits_to_save) + "f}").format(round(t + time_to_pass, digits_to_save)))
										
										#Checking prints continued
										#print('Time of arrival after change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Speed after', agent.speed)
										#print('NUM_BOYS', num_of_cars_before_agent_on_an_edge)
								#Break edge_num loop
								break
			#Delete variables from memory
			#del a, path_num

			#Agent finds path and expects to traverse it for some time Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					
					#If it is time to go - agent finds the path from source to target
					if a.times_of_departures[path_num] == t \
					and a.paths[path_num] == None and a.current_edge == (0,0):
						if a.PTA_user == True:
							a.paths[path_num] = self.dijkstra_PTA(source = a.start_nodes[path_num], 
																	target = a.finish_nodes[path_num], 
																	time = t, 
																	length = False)

						else:
							a.paths[path_num] = \
							dijkstra_with_time_dependent_weights(	graph = self.graph, 
																	source = a.start_nodes[path_num],
																	target = a.finish_nodes[path_num], 
																	agent = a, 
																	time = t, 
																	statistics = self.common_sense, 
																	length = False	)
						
						#To maintain expectation correctness we need to increase agent's speed to 60 kmh
						a.speed = 60
						#Then he finds the expected time he will be on each edge, also, expected travel time
						for edge_num in range(len(a.paths[path_num])-1):
							self.calculate_expected_edge_time_for_edge(agent = a,
																		path_num = path_num, 
																		edge_num = edge_num)

						if path_num <= len(a.paths)-2:
							for i in range(1,len(a.paths)-path_num):
								if a.expected_edge_times[path_num][-1][-1] >= a.times_of_departures[path_num + i]:
									a.times_of_departures[path_num+i] = float(a.expected_edge_times[path_num][-1][-1]) + 0.01 * i
									a.times_of_departures[path_num+i] = \
									float(("{0:." + str(len(str(self.agents[0].time_stamp)) - 2) + "f}")\
										.format(round(a.times_of_departures[path_num + i], digits_to_save)))
			
			#Delete variables from memory
			#del a, path_num

			#Agent comes to edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num])-1):
							
							#Agent comes to edge
							if a.real_edge_times[path_num][edge_num][0] == t and \
							a.current_edge == (0,0) and a.current_path == path_num:

								#Find the edge on which agent comes and increase the number of cars on it by one
								#Also, enqueue the agent on this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])
								num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name])
								agents_on_edges[edge_name].append(a)

								#Increase the daily number of cars came to the edge
								edge_name_came[edge_name] += 1

								#Recalculating speed and saving it in an ordered dict, also assigning 
								#values to agent's current variables
								a.current_edge = edge_name
								a.current_edge_num = edge_num
								a.current_path = path_num
								a.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, 
																agent = a)
								a.speed_by_time[t] = int(a.speed)
								a.edges_passed[path_num].append(edge_name)

								#Checking prints
								#print('----------COMING----------')
								#print('agent id', a.id_num)
								#print('number of cars before', num_of_cars_before_agent_on_an_edge)
								#print('time',t)
								#print('speed', a.speed)
								#print('edge', a.current_edge)

								#Agent estimates his time of arrival to the end of the edge taking into account 
								#current speed on edge for him
								self.calculate_real_edge_time_for_edge(agent = a, 
																		path_num = path_num, 
																		edge_num = edge_num)

							#Postponing departure times for next trips if they are earlier then expected
							#time of current path finish
							if path_num <= len(a.paths)-2 and a.real_edge_times[path_num][edge_num][-1] != None:
								for i in range(1,len(a.paths)-path_num):
									if a.real_edge_times[path_num][edge_num][-1] >= a.times_of_departures[path_num+i]:
										a.times_of_departures[path_num+i] = a.real_edge_times[path_num][edge_num][-1] + 0.01*i
										a.times_of_departures[path_num+i] = \
										float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) \
											+ "f}").format(round(a.times_of_departures[path_num+i], digits_to_save)))

			#Delete variables from memory
			#del a, path_num
			if arch_time == int((self.max_time/self.agents[0].time_stamp)):
				self.check_agents_prolong_simulation()
				#print(self.max_time)

		#Sorting speed_by_time dictionary for all agents by time
		self.sort_speed_by_time_for_all()

		#Test for consistency in simulation 
		#(number of cars came to edge must be equal to number of cars left the edge)
		#print("Equality:", edge_name_came == edge_name_left)
		if (edge_name_came == edge_name_left) == False:
			print('Equality violated')
			print('Came', edge_name_came)
			print('Left', edge_name_left)


	#Simulation optimized solution
	def simulation_optimized(self, max_iters = 3):
		#If proportion of PTA users is 1, then there is no need
		#for evolutionary optimization as equilibrium is deterministic
		#and will not change
		arch_start = time.time()
		#Preparation activities
		start = time.time()
		self.transform_historical_data()
		self.get_relationship_cars_ahead_speed()
		for a in self.agents:
			a.current_iter = len(self.common_sense)
		end = time.time()
		print('Preprocessing is done in', round((end - start), 2), 'sec')

		if self.proportion_users == 1 or self.proportion_users == 0:
			self.common_sense[len(self.common_sense)] = self.copy_agents()
			self.simulation_iteration_PTA()
			self.statistics[0] = self.copy_agents()
			self.common_sense[len(self.common_sense) - 1] = self.copy_agents()
			self.restore_all_agents()
			self.restore_all_agents_finally()
		
		#If proportion of navigator users is neither 1 nor 0
		#the 0-th iteration traffic situation is not an equilibrium
		# and thus needs to be evulutionary optimized in order to obtain
		#real equolibrium where not all drivers use navigator
		else:
			self.number_of_iterations = max_iters
			self.iteration = len(self.common_sense)
			for iteration in range(self.number_of_iterations):
				start = time.time()
				if iteration >= 1:
					self.historical_data = self.common_sense[self.iteration - 1]
					self.transform_historical_data()
				self.common_sense[self.iteration] = self.copy_agents()
				self.simulation_iteration_PTA()
				self.common_sense[self.iteration] = self.copy_agents()
				if iteration == self.number_of_iterations - 1:
					self.statistics[0] = self.copy_agents()
				self.restore_all_agents()
				self.paths_changed_from_last_iter()
				if self.paths_changed[self.iteration] == 0:
					self.iteration += 1
					self.number_of_iterations = self.iteration
					print('Agents paths converged to user-equilibrium')
					end = time.time()
					print('End of iteration', iteration, \
						'. Time elapsed:', round((end - start), 2), 'sec')
					self.statistics[0] = deepcopy(self.common_sense[len(self.common_sense) - 1])
					break
				self.iteration += 1
				end = time.time()
				print('End of iteration', iteration, \
					'. Time elapsed:', round((end - start), 2), 'sec')
			self.restore_all_agents_finally()
		arch_end = time.time()
		print('End of simulation', '. Time elapsed:', \
			round((arch_end - arch_start), 2), 'sec')


	def paths_changed_from_last_iter(self):
		if len(self.common_sense) - self.iteration >= 1:
			self.paths_changed[self.iteration] = 0
			for iteration in range(self.iteration, len(self.common_sense)):
				for a in self.common_sense[iteration]:
					for p in range(0,len(a.paths)):
						if a.paths[p] != self.common_sense[iteration-1]\
						[a.id_num].paths[p]:
							self.paths_changed[iteration] += 1
		else:
			pass


	#Get congestion data for edges
	def get_congestion_data_for_edges(self, iteration = 0):
		congestion = [dict((edge,0) for edge in self.graph.edges()) for t in \
		range(0,int(self.max_time/self.agents[0].time_stamp))]

		for t in range(0, int(self.max_time/self.agents[0].time_stamp)):
			idx = t
			t = float(t/100)

			for a in self.statistics[iteration]:
				for p in range(len(a.paths)):
					for edge_num in range(len(a.paths[p])-1):
						if t >= a.real_edge_times[p][edge_num][0] and \
						t < a.real_edge_times[p][edge_num][1]:
							edge_name = a.edges_passed[p][edge_num]
							congestion[idx][edge_name] += 1

		return congestion




class dynamic_system(simulation_baseline):
	def __init__(self, graph, agents, max_time, historical_data, common_sense, cars_ahead, all_three = True):
		simulation_baseline.__init__(self, graph, agents, 1, max_time)

		#Parameters initialization
		self.graph = graph
		self.agents = agents
		self.max_time = max_time
		self.statistics = dict()
		self.common_sense = common_sense
		del common_sense[len(common_sense)-1]
		self.num_baseline_iters = len(self.common_sense)

		self.historical_data = historical_data

		self.cars_ahead = cars_ahead
		self.min_speed = dict()
		for edge in self.cars_ahead.keys():
			self.min_speed[edge] = min(np.array(self.cars_ahead[edge])[:,0])
		self.max_speed = 60

		self.trips_data = dict((e, dict((t, [0, []]) \
			for t in range(0, int(max_time / 0.01) + 1))) \
			for e in self.graph.edges())
		self.choice_changed = dict()

		self.all_three = all_three

	#Function that updates weights of the edges for navigator
	def update_weights(self, agents_on_edges, current_time):
		num_pta_agents = 0
		num_all_agents = 0
		for edge_key, edge_list in agents_on_edges.items():
			distance_preserver = []
			update_list = []
			num_navi_agents = 0
			for agent in edge_list:
				num_all_agents += 1
				if agent.path_type[agent.current_path] == 'navigator':
					num_navi_agents += 1
				if agent.path_type[agent.current_path] == 'PTA':
					num_pta_agents += 1
			for agent in edge_list:
				if agent.path_type[agent.current_path] == 'navigator':
				
					passed_distance = self.calculate_passed_distance(agent = agent,
																	current_time = current_time)
					distance_to_pass = self.graph[edge_key[0]][edge_key[1]]['weight']\
					- passed_distance

					distance_preserver.append(distance_to_pass)
					if len(distance_preserver) == 1:
						update_val = (distance_preserver[-1] / agent.speed) * 60
					else:
						update_val = ((distance_preserver[-1] - \
									distance_preserver[-2]) / agent.speed) * 60
					update_list.append(update_val)
					if len(update_list) == num_navi_agents:
						update_list.append((passed_distance / agent.speed) * 60)
			if update_list != []:
				self.graph[edge_key[0]][edge_key[1]]['navigator_weight'] = \
				sum(update_list)
			#if current_time == 9.23:
				#print(self.graph[edge_key[0]][edge_key[1]]['navigator_weight'])
			
			del update_list
			del distance_preserver
		if num_all_agents == 0:
			num_all_agents = 1
		self.proportion_users = round(num_pta_agents / num_all_agents, 3)


	#Function that connects number of cars to speed (regression\AdaBoost)
	def get_relationship_cars_ahead_speed(self):
		data_sets = deepcopy(self.cars_ahead)
		self.reg = dict((e,0) for e in self.graph.edges())
		self.reg2 = dict((e,0) for e in self.graph.edges())
		for edge, data in data_sets.items():
			if data_sets[edge] == []:
				data_sets[edge].append([60, 0])
			data_sets[edge] = np.array(data_sets[edge])
			y = data_sets[edge][:,0].reshape(len(data_sets[edge]), 1).ravel()
			X = data_sets[edge][:,1].reshape(len(data_sets[edge][:,1]), 1)
			# Linear regression to predict speed depending on number of cars ahead
			#poly = PolynomialFeatures(degree = 3)
			#poly.fit_transform(X)
			#self.reg[edge] = linear_model.LinearRegression()
			#self.reg[edge].fit(X, y)
			# AdaBoost to predict speed depending on number of cars ahead
			self.reg2[edge] = AdaBoostRegressor(n_estimators = 50, 
												learning_rate = 1.0)
			self.reg2[edge].fit(X, y)
			#del self.cars_ahead


	#Function that transforms initial historical data about agents to the dict
	#which has number of cars on every edge at every time-momemt
	def transform_historical_data(self):
		self.cong_hist_data = dict((e, dict((t, [0, []]) for t in range(0, int(self.max_time / 0.01) + 1))) \
					for e in self.graph.edges())
		for a in self.historical_data:
			for p in range(len(a.paths)):
				for edge_num in range(len(a.paths[p])-1):
					start_time = int(round(a.real_edge_times[p][edge_num][0] / 0.01))
					finish_time = int(round(a.real_edge_times[p][edge_num][1] / 0.01))
					edge = a.edges_passed[p][edge_num]
					#time_range = finish_time - start_time
					for time in range(start_time, finish_time + 1):
						self.cong_hist_data[edge][time][0] += 1
						#If agent leaves the edge at time = t than it is indicated
						if time == finish_time:
							self.cong_hist_data[edge][time][1].append('l')


	#Function that calculates the subjective weight for given edge and time of
	#arrival using the data available for PTA server
	def calculate_weight_for_edge_time_diff(self, arrival_time, edge):
		min_speed = self.min_speed[edge]
		max_speed = 60
		time_stamp = self.agents[0].time_stamp
		weight_obj = 0
		weight_subj = 0
		t = int(round((arrival_time / time_stamp)))
		time_saved = int(round(arrival_time / time_stamp))

		position = round((self.trips_data[edge][t][0] * self.proportion_users) \
		+ (self.cong_hist_data[edge][t][0] * (1 - self.proportion_users)))
		left = round(len(self.trips_data[edge][t][1]) * self.proportion_users \
			+ len(self.cong_hist_data[edge][t][1]) * (1 - self.proportion_users))
		position -= left
		if position <= 0:
			position = 0
		position_saved = position

		speed = self.reg2[edge].predict(position_saved)
		if speed <= self.min_speed[edge]:
			speed = self.min_speed[edge]
		if speed >= self.max_speed:
			speed = self.max_speed

		while weight_obj < self.graph[edge[0]][edge[1]]['weight']:
			
			while position == position_saved:
				t += 1
				if t >= int(round(self.max_time / 0.01)):
					break
				left = round(len(self.trips_data[edge][t][1]) * self.proportion_users \
					+ len(self.cong_hist_data[edge][t][1]) * (1 - self.proportion_users))
				position -= left
				if position <= 0:
					position = 0
			
			time_passed = t - time_saved
			time_saved = t

			weight_obj += round(float(speed * (time_passed * time_stamp)), \
				len(str(time_stamp)) - 2)
			weight_subj += round(60 * (time_passed * time_stamp), \
				len(str(time_stamp)) - 2)

			speed = self.reg2[edge].predict(position)
			if speed <= self.min_speed[edge]:
				speed = self.min_speed[edge]
			if speed >= self.max_speed:
				speed = self.max_speed

			position_saved = position


		if weight_obj >= self.graph[edge[0]][edge[1]]['weight']:
			diff = weight_obj - self.graph[edge[0]][edge[1]]['weight']
			time_diff = diff / speed
			weight_obj -= diff
			weight_subj -= time_diff * 60
			return round(float(weight_subj), len(str(time_stamp)) - 2)

		return round(float(weight_subj), len(str(time_stamp)) - 2)


	# Classical Dijkstra algorithm which uses time-dependent weights
	# supplied by PTA server
	def dijkstra_PTA(self, source, target, time, length = False):
	
		digits_to_save = len(str(self.agents[0].time_stamp)) - 2
		
		adj_list = self.graph.adjacency_list()

		dist = [float('inf')]*len(adj_list)
		prev = [None]*len(adj_list)
		dist[source-1] = 0
		prev[source-1] = source - 1

		dist_proxy = [(0, source - 1)]
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
				time = float(("{0:." + str(digits_to_save) + "f}").format \
					(round(init_time + (dist[min_node] / 60), digits_to_save)))
				#print(time)
			
			for vertex in adj_list[min_node]:
				vertex = vertex-1
				cost = self.calculate_weight_for_edge_time_diff(arrival_time = time, 
															edge = (min_node+1, vertex+1))
					
				if dist[vertex] > dist[min_node] + cost:
					proxy_var = dist[vertex]
					dist[vertex] = dist[min_node] + cost
					heappush(dist_proxy, (dist[vertex], vertex))
					prev[vertex] = min_node


	# Function that counts the number of paths that were changed from previous
	# to current iteration
	def choose_type_of_path(self, agent, path_num, t):
		# Count the lengths of all 3 available options 
		# and subtract the percieved distrust (measured in 
		# prolongation of the road) associated with the 
		# method
		baseline_path_len = \
		dijkstra_with_time_dependent_weights(	graph = self.graph, 
												source = agent.start_nodes[path_num],
												target = agent.finish_nodes[path_num],
												agent = agent,
												time = t,
												statistics = self.common_sense,
												length = True	) \
		+ (self.calculate_discounted_trust(	agent = agent, 
											path_num = path_num, 
											method = 'baseline'	) * 60)
		if self.all_three == True:
			PTA_path_len = \
			self.dijkstra_PTA(	source = agent.start_nodes[path_num], 
								target = agent.finish_nodes[path_num], 
								time = t, 
								length = True	) \
			+ (self.calculate_discounted_trust(	agent = agent, 
												path_num = path_num, 
												method = 'PTA'	) * 60)
		elif self.all_three == False:
			PTA_path_len = 100000

		navigator_path_len = \
		nx.dijkstra_path_length(	G = self.graph, 
									source = agent.start_nodes[path_num], 
									target = agent.finish_nodes[path_num], 
									weight = 'navigator_weight'	) \
		+ (self.calculate_discounted_trust(	agent = agent, 
											path_num = path_num, 
											method = 'navigator'	) * 60)

		# Choose one or many which are best in terms of length
		min_path_len = min(baseline_path_len, PTA_path_len, navigator_path_len)
		# Baseline is the single best one
		if min_path_len == baseline_path_len \
		and min_path_len != navigator_path_len \
		and min_path_len != PTA_path_len:
			path = dijkstra_with_time_dependent_weights(	graph = self.graph, 
															source = agent.start_nodes[path_num],
															target = agent.finish_nodes[path_num],
															agent = agent, 
															time = t, 
															statistics = self.common_sense, 
															length = False	)
			path_type = 'baseline'
		# Navigator is the single best one
		elif min_path_len == navigator_path_len \
		and min_path_len != baseline_path_len \
		and min_path_len != PTA_path_len:
			path = nx.dijkstra_path(	G = self.graph, 
										source = agent.start_nodes[path_num], 
										target = agent.finish_nodes[path_num], 
										weight = 'navigator_weight'	)
			path_type = 'navigator'
		# PTA is the single best one
		elif min_path_len == PTA_path_len \
		and min_path_len != navigator_path_len \
		and min_path_len != baseline_path_len:
			path = self.dijkstra_PTA(	source = agent.start_nodes[path_num], 
										target = agent.finish_nodes[path_num], 
										time = t, 
										length = False	)
			path_type = 'PTA'
		# Baseline and navigator are best ones
		elif min_path_len == baseline_path_len \
		and min_path_len == navigator_path_len \
		and min_path_len != PTA_path_len:
			baseline_path = dijkstra_with_time_dependent_weights(	graph = self.graph, 
																	source = agent.start_nodes[path_num],
																	target = agent.finish_nodes[path_num], 
																	agent = agent, 
																	time = t, 
																	statistics = self.common_sense, 
																	length = False	)
			navigator_path = nx.dijkstra_path(	G = self.graph, 
												source = agent.start_nodes[path_num], 
												target = agent.finish_nodes[path_num], 
												weight = 'navigator_weight'	)
			choice_set = (baseline_path, navigator_path)
			path = choice_set[np.random.choice((len(choice_set)), p = [0.5, 0.5])]
			if path == baseline_path:
				path_type = 'baseline'
			elif path == navigator_path:
				path_type = 'navigator'
		# Baseline and PTA are best ones
		elif min_path_len == baseline_path_len \
		and min_path_len == PTA_path_len \
		and min_path_len != navigator_path_len:
			baseline_path = dijkstra_with_time_dependent_weights(	graph = self.graph, 
																	source = agent.start_nodes[path_num],
																	target = agent.finish_nodes[path_num], 
																	agent = agent, 
																	time = t, 
																	statistics = self.common_sense, 
																	length = False	)
			PTA_path = self.dijkstra_PTA(	source = agent.start_nodes[path_num], 
											target = agent.finish_nodes[path_num], 
											time = t, 
											length = False	)
			choice_set = (baseline_path, PTA_path)
			path = choice_set[np.random.choice((len(choice_set)), p = [0.5, 0.5])]
			if path == baseline_path:
				path_type = 'baseline'
			elif path == PTA_path:
				path_type = 'PTA'
		# Navigator and PTA are best ones
		elif min_path_len == navigator_path_len \
		and min_path_len == PTA_path_len \
		and min_path_len != baseline_path_len:
			navigator_path = nx.dijkstra_path(	G = self.graph, 
												source = agent.start_nodes[path_num], 
												target = agent.finish_nodes[path_num], 
												weight = 'navigator_weight'	)
			PTA_path = self.dijkstra_PTA(	source = agent.start_nodes[path_num], 
											target = agent.finish_nodes[path_num], 
											time = t, 
											length = False	)
			choice_set = (navigator_path, PTA_path)
			path = choice_set[np.random.choice((len(choice_set)), p = [0.5, 0.5])]
			if path == navigator_path:
				path_type = 'navigator'
			elif path == PTA_path:
				path_type = 'PTA'
		# All three are equally great
		elif min_path_len == navigator_path_len \
		and min_path_len == PTA_path_len \
		and min_path_len == baseline_path_len:
			baseline_path = dijkstra_with_time_dependent_weights(	graph = self.graph, 
																	source = agent.start_nodes[path_num],
																	target = agent.finish_nodes[path_num], 
																	agent = agent, 
																	time = t, 
																	statistics = self.common_sense, 
																	length = False	)
			navigator_path = nx.dijkstra_path(	G = self.graph, 
												source = agent.start_nodes[path_num], 
												target = agent.finish_nodes[path_num], 
												weight = 'navigator_weight'	)
			PTA_path = self.dijkstra_PTA(	source = agent.start_nodes[path_num], 
											target = agent.finish_nodes[path_num], 
											time = t, 
											length = False	)
			choice_set = (baseline_path, navigator_path, PTA_path)
			path = choice_set[np.random.choice((len(choice_set)), p = [1/3, 1/3, 1/3])]
			if path == baseline_path:
				path_type = 'baseline'
			elif path == navigator_path:
				path_type = 'navigator'
			elif path == PTA_path:
				path_type = 'PTA'

		return path, path_type


	# Expected edge times
	def calculate_expected_edge_time_for_edge(self, agent, path_num, edge_num):
		digits_to_save = len(str(agent.time_stamp)) - 2
		if edge_num == 0:
			#Expected edge time is calculated as the edge subjective (for agent) 
			#weight divided by 60 kmh (default speed)
			#Expected edge time is a time in which agent expects to finish traversing the edge
			#Expected times are different for those who use navigator and those who don't
			if agent.path_type[path_num] == 'PTA':
				time = agent.times_of_departures[path_num]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				time_range = \
				self.calculate_weight_for_edge_time_diff(arrival_time = time, 
															edge = edge) / agent.speed
				time_range = float(("{0:." + str(digits_to_save) \
				+ "f}").format(round(time_range, digits_to_save)))

				agent.expected_edge_times[path_num].append(list((time, time + time_range)))
				#Update trips data on the PTA server
				for t in range(int(round(time / agent.time_stamp)), \
					int(round((time + time_range) / agent.time_stamp)) + 1):
					self.trips_data[edge][t][0] += 1
					#If agent leaves the edge at time = t than it is indicated
					if t == int(round((time + time_range) / agent.time_stamp)):
						self.trips_data[edge][t][1].append('l')
			elif agent.path_type[path_num] == 'baseline':
				time = agent.times_of_departures[path_num]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (calculate_subjective_weight_for_edge_time_overall(graph = self.graph,
																		agent = agent,
																		time = time,
																		edge = edge,
																		statistics = self.common_sense) \
				/ agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			elif agent.path_type[path_num] == 'navigator':
				time = agent.times_of_departures[path_num]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (self.graph[edge[0]][edge[1]]['navigator_weight'] / agent.speed))))
			agent.real_edge_times[path_num].append(list((time, None)))
		
		else:
			if agent.path_type[path_num] == 'PTA':
				time = agent.expected_edge_times[path_num][edge_num-1][1]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				time_range = self.calculate_weight_for_edge_time_diff(arrival_time = time, 
																		edge = edge) / agent.speed
				time_range = float(("{0:." + str(digits_to_save) \
				+ "f}").format(round(time_range, digits_to_save)))
				agent.expected_edge_times[path_num].append(list((time, time + time_range)))
				#if agent.id_num == 531 and edge == (1,5):
				#	print(time + time_range)
				#	print(self.calculate_weight_for_edge_time_diff(arrival_time = time, 
				#														edge = edge))
				#Update trips data on the PTA server
				for t in range(int(round(time / agent.time_stamp)), \
					int(round((time + time_range) / agent.time_stamp)) + 1):
					self.trips_data[edge][t][0] += 1
					#If agent leaves the edge at time = t than it is indicated
					if t == int(round((time + time_range) / agent.time_stamp)):
						self.trips_data[edge][t][1].append('l')
			elif agent.path_type[path_num] == 'baseline':
				time = agent.expected_edge_times[path_num][edge_num-1][1]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num + 1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (calculate_subjective_weight_for_edge_time_overall(graph = self.graph, \
																		agent = agent, 
																		time = time,
																		edge = edge,
																		statistics = self.common_sense) \
				/ agent.speed))))
			#Real edge times modified below to copy the expected edge times without reference
			elif agent.path_type[path_num] == 'navigator':
				time = agent.expected_edge_times[path_num][edge_num-1][1]
				edge = (agent.paths[path_num][edge_num], agent.paths[path_num][edge_num+1])
				agent.expected_edge_times[path_num].append(list((time, time \
				+ (self.graph[edge[0]][edge[1]]['navigator_weight'] / agent.speed))))
			agent.real_edge_times[path_num].append(list((None, None)))
		#Everything is rounded to maintain discrete time consistency
		for fl in range(2):
			agent.expected_edge_times[path_num][edge_num][fl] = \
			float(("{0:." + str(digits_to_save) + \
				"f}").format(round(agent.expected_edge_times[path_num][edge_num][fl], digits_to_save)))
		agent.real_edge_times[0][0][0] = \
		float(("{0:." + str(digits_to_save) + \
			"f}").format(round(agent.real_edge_times[0][0][0], digits_to_save)))


	# Function that performs one run of simulation iteration
	def simulation_iteration_DS(self):

		digits_to_save = len(str(self.agents[0].time_stamp)) - 2

		#Priorty qeues for every edge and counters for each edge to check whether every agent who
		#came to edge also left it
		agents_on_edges = dict((e,[]) for e in self.graph.edges())
		edge_name_came = dict((e, 0) for e in self.graph.edges())
		edge_name_left = dict((e, 0) for e in self.graph.edges())

		#Main Loop that makes each iteration of the loop tick with the time interval specified by time_stamp
		#for i in range(0,int((self.max_time/self.agents[0].time_stamp))):
		#	t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
		#		"f}").format(round((0 + i*self.agents[0].time_stamp), digits_to_save)))
		arch_time = -1
		while arch_time != int((self.max_time/self.agents[0].time_stamp)):
			arch_time += 1
			t = float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) + \
				"f}").format(round((0 + arch_time*self.agents[0].time_stamp), digits_to_save)))

			#Agent leaves the edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num])-1):

							#Agent leaves the edge
							if a.real_edge_times[path_num][edge_num][1] == t:

								#Find the edge from which agent leaves and decrease the number of cars on it by one
								#Also, dequeue agent from this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])
								index = agents_on_edges[edge_name].index(a)
								agents_on_edges[edge_name].pop(index)
								#if index != 0:
									#print('Time', t, 'Agent', a.id_num, 'IDX', index)
								#	agents_on_edges[edge_name].pop(index)
								#if index == 0:
								#	agents_on_edges[edge_name].pop(0)

								#Check
								edge_name_left[edge_name] += 1

								#Checking prints
								#print('----------LEAVING----------')
								#print('agent id', a.id_num)
								#print('number_of_cars_after', len(agents_on_edges[edge_name]))
								#print('time',t)
								#print('edge',edge_name)

								#Cleans current edge variables and transfers time of edge-finish to next edge start time
								#Or in case this is the end the path for agent increases his current path number by one
								a.current_edge = (0,0)
								a.current_edge_num = []
								if edge_num <= (len(a.real_edge_times[path_num])-2):
									a.real_edge_times[path_num][edge_num+1][0] = \
									a.real_edge_times[path_num][edge_num][1]
								if edge_num == len(a.real_edge_times[path_num])-1:
									a.current_path += 1
									diff_real = (a.real_edge_times[path_num][edge_num][-1] \
										- a.real_edge_times[path_num][0][0])
									diff_exp = (a.expected_edge_times[path_num][edge_num][-1] \
										- a.expected_edge_times[path_num][0][0])
									a.diff_exp_real[path_num] = diff_real - diff_exp

								
								#Recalculating real travel times for every agent on an edge 
								#(who is not leacing at the current time) from which another agent left
								#Their speed must change due to decrese in number of cars ahead of them
								cntr = -1
								for agent in agents_on_edges[edge_name]:
									cntr += 1
									if agent.current_edge == edge_name and \
									(agent.real_edge_times[agent.current_path][agent.current_edge_num][1] != t):

										#Searching for the position of the agent on an edge and finding number of cars ahead
										num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name][:int(cntr)])

										#Recalculating speed and saving it in an ordered dict
										agent.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, agent = agent)
										agent.speed_by_time[t] = int(agent.speed)

										#Calculating the distance on edge which is already passed by an agent, distance to pass,
										#and finally updating the real travel time on the edge
										passed_distance = self.calculate_passed_distance(agent = agent, 
																							current_time = t)

										weight_of_the_edge = self.graph[edge_name[0]][edge_name[1]]['weight']
										distance_to_pass = weight_of_the_edge - passed_distance

										#Checking prints
										#print('----------------')
										#print('Time', t)
										#print('agent id', agent.id_num)
										#print('Path', agent.current_path)
										#print("Current edge", agent.current_edge)
										#print('Time of arrival before change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Passed Distance', passed_distance)

										time_to_pass = float(("{0:." + str(digits_to_save) \
											+ "f}").format(distance_to_pass / agent.speed))
										#Small fix to secure problems with rounding
										if time_to_pass == 0.0:
											time_to_pass = 0.01
										agent.real_edge_times[agent.current_path][agent.current_edge_num][1] = \
										float(("{0:." + str(digits_to_save) + "f}").format(round(t + time_to_pass, digits_to_save)))
										
										#Checking prints continued
										#print('Time of arrival after change', agent.real_edge_times[agent.current_path][agent.current_edge_num][1])
										#print('Speed after', agent.speed)
										#print('NUM_BOYS', num_of_cars_before_agent_on_an_edge)
								#Break edge_num loop
								break
			#Delete variables from memory
			#del a, path_num

			#Update navigator weights
			self.update_weights(agents_on_edges = agents_on_edges,
								current_time = t)

			#Agent finds path and expects to traverse it for some time Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					
					#If it is time to go - agent finds the path from source to target
					if a.times_of_departures[path_num] == t \
					and a.paths[path_num] == None and a.current_edge == (0,0):
						
						a.paths[path_num], a.path_type[path_num] = \
						self.choose_type_of_path(	agent = a, 
													path_num = path_num, 
													t = t	)
						
						#To maintain expectation correctness we need to increase agent's speed to 60 kmh
						a.speed = 60
						#Then he finds the expected time he will be on each edge, also, expected travel time
						for edge_num in range(len(a.paths[path_num])-1):
							self.calculate_expected_edge_time_for_edge(	agent = a,
																		path_num = path_num, 
																		edge_num = edge_num)

						if path_num <= len(a.paths)-2:
							for i in range(1,len(a.paths)-path_num):
								if a.expected_edge_times[path_num][-1][-1] >= a.times_of_departures[path_num + i]:
									a.times_of_departures[path_num+i] = float(a.expected_edge_times[path_num][-1][-1]) + 0.01 * i
									a.times_of_departures[path_num+i] = \
									float(("{0:." + str(len(str(self.agents[0].time_stamp)) - 2) + "f}")\
										.format(round(a.times_of_departures[path_num + i], digits_to_save)))
			
			#Delete variables from memory
			#del a, path_num

			#Agent comes to edge Arch-loop
			for a in self.agents:
				#for path_num in range(len(a.paths)):
				for path_num in range(a.current_path, len(a.paths)):
					if a.paths[path_num] != None:
						for edge_num in range(len(a.paths[path_num])-1):
							
							#Agent comes to edge
							if a.real_edge_times[path_num][edge_num][0] == t and \
							a.current_edge == (0,0) and a.current_path == path_num:

								#Find the edge on which agent comes and increase the number of cars on it by one
								#Also, enqueue the agent on this edge
								edge_name = tuple(a.paths[path_num][edge_num:edge_num+2])
								num_of_cars_before_agent_on_an_edge = len(agents_on_edges[edge_name])
								agents_on_edges[edge_name].append(a)

								#Increase the daily number of cars came to the edge
								edge_name_came[edge_name] += 1

								#Recalculating speed and saving it in an ordered dict, also assigning 
								#values to agent's current variables
								a.current_edge = edge_name
								a.current_edge_num = edge_num
								a.current_path = path_num
								a.speed = self.speed_revisited(num_of_cars_before_agent_on_an_edge, 
																agent = a)
								a.speed_by_time[t] = int(a.speed)
								a.edges_passed[path_num].append(edge_name)

								#Checking prints
								#print('----------COMING----------')
								#print('agent id', a.id_num)
								#print('number of cars before', num_of_cars_before_agent_on_an_edge)
								#print('time',t)
								#print('speed', a.speed)
								#print('edge', a.current_edge)

								#Agent estimates his time of arrival to the end of the edge taking into account 
								#current speed on edge for him
								self.calculate_real_edge_time_for_edge(agent = a, 
																		path_num = path_num, 
																		edge_num = edge_num)

							#Postponing departure times for next trips if they are earlier then expected
							#time of current path finish
							if path_num <= len(a.paths)-2 and a.real_edge_times[path_num][edge_num][-1] != None:
								for i in range(1,len(a.paths)-path_num):
									if a.real_edge_times[path_num][edge_num][-1] >= a.times_of_departures[path_num+i]:
										a.times_of_departures[path_num+i] = a.real_edge_times[path_num][edge_num][-1] + 0.01*i
										a.times_of_departures[path_num+i] = \
										float(("{0:." + str(len(str(self.agents[0].time_stamp))-2) \
											+ "f}").format(round(a.times_of_departures[path_num+i], digits_to_save)))

			#Delete variables from memory
			#del a, path_num
			if arch_time == int((self.max_time/self.agents[0].time_stamp)):
				self.check_agents_prolong_simulation()
				#print(self.max_time)
			#print(t)

		#Sorting speed_by_time dictionary for all agents by time
		self.sort_speed_by_time_for_all()

		#Test for consistency in simulation 
		#(number of cars came to edge must be equal to number of cars left the edge)
		#print("Equality:", edge_name_came == edge_name_left)
		if (edge_name_came == edge_name_left) == False:
			print('Equality violated')
			print('Came', edge_name_came)
			print('Left', edge_name_left)


	# Simulation optimized solution
	def simulation_optimized(self, max_iters = 5):
		# If proportion of PTA users is 1, then there is no need
		# for evolutionary optimization as equilibrium is deterministic
		# and will not change
		arch_start = time.time()
		# Preparation activities
		start = time.time()
		self.transform_historical_data()
		self.get_relationship_cars_ahead_speed()
		for a in self.agents:
			a.current_iter = len(self.common_sense)
		end = time.time()
		print('Preprocessing is done in', round((end - start), 2), 'sec')
		self.number_of_iterations = max_iters
		self.iteration = len(self.common_sense)
		for iteration in range(self.number_of_iterations):
			start = time.time()
			if iteration >= 1:
				self.historical_data = self.common_sense[self.iteration -1]
				self.transform_historical_data()
			self.common_sense[self.iteration] = self.copy_agents()
			self.simulation_iteration_DS()
			self.common_sense[self.iteration] = self.copy_agents()
			self.check_method_stability()
			if iteration == self.number_of_iterations - 1:
				self.statistics[0] = self.copy_agents()
			self.restore_all_agents()
			if self.choice_changed[self.iteration] == 0:
				self.iteration += 1
				self.number_of_iterations = self.iteration
				print('Agents paths converged to user-equilibrium')
				end = time.time()
				print('End of iteration', iteration, \
					'. Time elapsed:', round((end - start), 2), 'sec')
				self.statistics[0] = deepcopy(self.common_sense[len(self.common_sense) - 1])
				break
			self.iteration += 1
			end = time.time()
			print('End of iteration', iteration, \
				'. Time elapsed:', round((end - start), 2), 'sec')
		self.restore_all_agents_finally()
		arch_end = time.time()
		print('End of simulation', '. Time elapsed:', \
			round((arch_end - arch_start), 2), 'sec')


	#CHECK IT IN TERMS OF PATH NUM
	# Function that claculates the weighted average difference of the forecast
	# of the method used for current_path and reality
	def calculate_discounted_trust(self, agent, path_num, method):
		memory_list = []
		for iteration in range(self.num_baseline_iters, len(self.common_sense)):
			if iteration == len(self.common_sense) - 1:
				for item in self.agents[agent.id_num].path_type[:path_num+1]:
					if item == method:
						diff = \
						self.agents[agent.id_num].diff_exp_real\
						[self.agents[agent.id_num].path_type[:path_num+1].index(item)]
						memory_list.append(diff)
						#print('Diff', diff, 'Agent', agent.id_num, 'item', item)

			else:
				for item in self.common_sense\
					[iteration][agent.id_num].path_type[:path_num+1]:
					if item == method:
						diff = \
						self.common_sense[iteration][agent.id_num].diff_exp_real\
						[self.common_sense\
						[iteration][agent.id_num].path_type[:path_num+1].index(item)]
						memory_list.append(diff)
						#print('Diff', diff, 'Agent', agent.id_num, 'item', item)
		discount = get_discount(len(memory_list))
		if discount == []:
			discounted_diff = 0
		else:
			discounted_diff = 0
			for w in range(len(discount)):
				discounted_diff += memory_list[w] * discount[w]
		return discounted_diff


	# Function that checks whether every agent sticks to his choice of method
	# (for a given path) for at least 2 iterations
	def check_method_stability(self):
		if len(self.common_sense) - self.iteration >= 1:
			self.choice_changed[self.iteration] = 0
			for iteration in range(self.iteration, len(self.common_sense)):
				for a in self.common_sense[iteration]:
					for p in range(0,len(a.paths)):
						if a.path_type[p] != self.common_sense[iteration-1]\
						[a.id_num].path_type[p]:
							self.choice_changed[iteration] += 1
		else:
			pass







