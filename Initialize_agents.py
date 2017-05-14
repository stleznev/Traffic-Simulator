import random as rd
import numpy as np
import networkx as nx

#Class that defines what agent is and initializes it
#Comments for me: 
#Parameters of gauss distribution need to be tuned
#Check probability parameter in home/random last point
#Make more sane the part with departure times
#Everything is rounded to 2 digit after integer
class Agent(object):
	def __init__(self, id_num, speed, graph, time_stamp, navi_user_prob = 1, PTA_user_prob = 1):

		self.digits_to_round = len(str(time_stamp)) - 2

		self.speed = speed
		self.id_num = id_num
		self.time_stamp = time_stamp

		self.num_trips = abs(int(rd.gauss(2,3)))
		while self.num_trips == 0:
			self.num_trips = abs(int(rd.gauss(2,3)))

		#THINK HOW TO MAKE NO SINGLE-POINT PATHS
		self.start_nodes = [None] * self.num_trips
		self.finish_nodes = [None] * self.num_trips
		for i in range(self.num_trips):
			if i == 0:
				nodes = graph.nodes()
				first_dep_point = rd.choice(nodes)
				nodes.remove(first_dep_point)
				first_arr_point = rd.choice(nodes)
				while first_dep_point == first_arr_point:
					nodes = graph.nodes()
					first_dep_point = rd.choice(nodes)
					nodes.remove(first_dep_point)
					first_arr_point = rd.choice(nodes)
				self.start_nodes[i] = first_dep_point
				self.finish_nodes[i] = first_arr_point
				nodes = graph.nodes()
			elif i == self.num_trips-1:
				self.start_nodes[i] = self.finish_nodes[i-1]
				nodes.remove(self.finish_nodes[i-1])
				self.finish_nodes[i] = \
				np.random.choice(	(rd.choice(nodes),first_dep_point), 
									p = [0.2, 0.8]	)
				while self.start_nodes[i] == self.finish_nodes[i]:
					self.finish_nodes[i] = rd.choice(nodes)
			else:
				self.start_nodes[i] = self.finish_nodes[i-1]
				nodes.remove(self.finish_nodes[i-1])
				self.finish_nodes[i] = rd.choice(nodes)
				nodes = graph.nodes()
		del nodes
		del first_dep_point
		del first_arr_point

		#Think how to deal with late departures/late arrivals (time >= 24)
		#Postpone departures well!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		#18, 23 - was
		#27, 100 now
		self.times_of_departures = [None]*self.num_trips
		first_dep = round(abs(rd.gauss(8,2)),self.digits_to_round)
		last_dep = round(abs(rd.gauss(18,2)),self.digits_to_round)
		while first_dep >= float(23) or last_dep >= float(23):
			first_dep = round(abs(rd.gauss(8,2)),self.digits_to_round)
			last_dep = round(abs(rd.gauss(18,2)),self.digits_to_round)
		while first_dep >= last_dep:
			first_dep = round(abs(rd.gauss(8,2)),self.digits_to_round)
			last_dep = round(abs(rd.gauss(18,2)),self.digits_to_round)
		for i in range(self.num_trips):
			if i == 0:
				self.times_of_departures[i] = float(('%.' + str(self.digits_to_round) + 'f') \
					% round(first_dep, self.digits_to_round))
			elif i == self.num_trips-1:
				self.times_of_departures[i] = float(('%.' + str(self.digits_to_round) + 'f') \
					% round(last_dep, self.digits_to_round))

			else:
				self.times_of_departures[i] = float(('%.' + str(self.digits_to_round) + 'f') \
					% round(self.times_of_departures[0] + i*(last_dep - first_dep)/(self.num_trips-1), self.digits_to_round))
				self.times_of_departures[i] = float(("{0:." + str(self.digits_to_round) + \
				"f}").format(round((self.times_of_departures[0] + i*(last_dep - first_dep)/(self.num_trips-1)), self.digits_to_round)))

		del first_dep
		del last_dep

		self.original_times_of_departures = list(self.times_of_departures)
		self.paths = [None] * self.num_trips

		self.expected_edge_times = [[] for t in range(self.num_trips)]
		self.real_edge_times = [[] for t in range(self.num_trips)]

		self.current_edge = (0,0)
		self.current_path = 0
		self.current_edge_num = []

		self.speed_by_time = dict()

		self.edges_passed = [[] for t in range(self.num_trips)]

		self.current_iter = 0

		#Initialize agent to use or not to use navigator application
		self.navigator_user = \
		np.random.choice(	(True, False), 
							p = [navi_user_prob, 1 - navi_user_prob]	)
		#Initialize agent to use or not to use PTA application
		self.PTA_user = \
		np.random.choice(	(True, False), 
							p = [PTA_user_prob, 1 - PTA_user_prob]	)

		self.path_type = [None] * self.num_trips
		self.diff_exp_real = [None] * self.num_trips


	#Function that removes trips that begin/end after maximum allowed
	#by simulation time
	def refuse_trips(self, path_num, max_time):
		if self.expected_edge_times[path_num][-1][-1] >= max_time:
		#or self.times_of_departures[path_num] >= max_time:
			for trip in range(path_num, self.num_trips):
				self.num_trips -= 1
				self.start_nodes.pop(-1)
				self.finish_nodes.pop(-1)
				self.paths.pop(-1)
				self.real_edge_times.pop(-1)
				self.expected_edge_times.pop(-1)
				self.times_of_departures.pop(-1)
				self.original_times_of_departures.pop(-1)
				self.edges_passed.pop(-1)

				self.current_edge = (0,0)
				self.current_path = 0
				self.current_edge_num = []


	#Fucntion that restores agent to his pre-simulation state
	def restore(self, finish = False):
		self.speed = 60
		self.paths = [None]*self.num_trips
		self.expected_edge_times = [[] for t in range(self.num_trips)]
		self.real_edge_times = [[] for t in range(self.num_trips)]
		self.edges_passed = [[] for t in range(self.num_trips)]
		self.speed_by_time = dict()
		self.times_of_departures = list(self.original_times_of_departures)

		self.current_edge = (0,0)
		self.current_path = 0
		self.current_edge_num = []
		self.current_iter += 1

		if finish == True:
			self.current_iter = 0



#Function that creates a list of agents (as class objects)
def create_list_of_agents(num_agents, graph, time_stamp, navi_user_prob = 1, PTA_user_prob = 1):
	agents = []
	for i in range(num_agents):
		agents.append(Agent(id_num = i, 
							speed = 60, 
							graph = graph, 
							time_stamp = time_stamp, 
							navi_user_prob = navi_user_prob, 
							PTA_user_prob = PTA_user_prob))
	return agents





