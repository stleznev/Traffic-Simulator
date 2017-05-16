# Check
# No non-traversed paths
# No different departure times (expected vs real)
# No departure before last trip's arrival
def check_paths_and_travel_times(simulation):
	for iteration in range(len(simulation.common_sense)):
		for a in simulation.common_sense[iteration]:
			for p in range(len(a.paths)):
				if a.paths[p] == None:
					print('Broken paths')
					print('Iteration', iteration, 'ID', a.id_num, 'Path', p)
				else:
					for edge in range(len(a.paths[p])-1):
						if a.real_edge_times[p][edge][0] == None or a.real_edge_times[p][edge][-1] == None:
							print('---------------------------------------------------')
							print('Error in real_edge_times assignment: Iteration', iteration) 
							print('ID', a.id_num, 'Path', p)

						if a.expected_edge_times[p][0][0] != a.real_edge_times[p][0][0]:
							print('--------------------------------------------------------------------------------------')
							print('Different expected and real times: Iteration', iteration)
							print('ID', a.id_num, 'Path', p)

			for path_num in range(1,len(a.paths)):
				if a.paths[path_num] != None:
					if a.real_edge_times[path_num][0][0] < a.real_edge_times[path_num-1][-1][-1] or\
						a.expected_edge_times[path_num][0][0] < a.expected_edge_times[path_num-1][-1][-1]:
							print('--------------------------------------------------------------------------------------')
							print('Error, broken times of departures: Iteration', iteration)
							print('ID', a.id_num, 'Path', path_num)
					else:
						pass


# FIFO property check
# Takes a lot of time, run only if necessary
def check_FIFO_property_baseline(simulation):
	for iteration in range(len(simulation.common_sense)):
		for a in simulation.common_sense[iteration]:
			for p in range(len(a.paths)):
				for edge in a.edges_passed[p]:
					check_list = [G[edge[0]][edge[1]]['weight']]
					for i in range(0,simulation.max_time*100):
						t = float(i/100)
						check_list.append(calculate_subjective_weight_for_edge_time(agent = a, \
																					time= t , edge = edge, \
																					iteration = iteration, \
																					statistics = simulation.common_sense))
						#print('ID', a.id_num, 'path', p, 'edge', edge, 'time', t)
						#print(check_list[-1])
						if t + check_list[-1]/60 <= round(t - 0.01, 2) + check_list[-2]/60:
							print('FIFO is not satisfied')
							print('ID',a.id_num,'Path', p, 'Edge', edge, 'Time', t)
							break


# Function that compares expectations of agents on the travel times on
# each edge of their path to the least possible time needed to traverse it
def check_expectations(simulation):
	index = len(simulation.statistics) - 1
	for a in simulation.statistics[index]:
		for p in range(len(a.paths)):
			for edge_num in range(len(a.paths[p]) - 1):
				trav_time_exp = a.expected_edge_times[p][edge_num][-1] - a.expected_edge_times[p][edge_num][0]
				wght = simulation.graph[a.edges_passed[p][edge_num][0]][a.edges_passed[p][edge_num][-1]]['weight'] / 60
				if round(wght, 2) > round(trav_time_exp, 2):
					print(simulation)
					print('ID', a.id_num, 'Path', p, 'Min time', wght, 'Trav exp', round(trav_time_exp, 2))


# Function that compares reality of the travel times on
# each edge of agents' paths to the least possible time needed to traverse it
def check_reality(simulation):
	index = len(simulation.statistics) - 1
	for a in simulation.statistics[index]:
		for p in range(len(a.paths)):
			for edge_num in range(len(a.paths[p]) - 1):
				trav_time_real = a.real_edge_times[p][edge_num][-1] - a.real_edge_times[p][edge_num][0]
				wght = simulation.graph[a.edges_passed[p][edge_num][0]][a.edges_passed[p][edge_num][-1]]['weight'] / 60
				if round(wght, 2) > round(trav_time_real, 2):
					print(simulation)
					print('ID', a.id_num, 'Path', p, 'Min time', wght, 'Trav real', round(trav_time_real, 2))


# FIFO property check
# Takes a lot of time, run only if necessary
def check_FIFO_property_PTA(simulation):
	for edge in G.edges():
		check_list = [G[edge[0]][edge[1]]['weight']]
		for i in range(0,25*100):
			t = float(i/100)
			check_list.append(simulation.calculate_weight_for_edge_time_diff(	arrival_time = t,
																			edge = edge))
			if t + check_list[-1]/60 <= round(t - 0.01, 2) + check_list[-2]/60:
				print('FIFO is not satisfied')
				print('ID',a.id_num,'Path', p, 'Edge', edge, 'Time', t)
				break




