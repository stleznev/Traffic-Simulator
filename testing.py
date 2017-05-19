def check_paths_and_travel_times(simulation):
    """Function that checks for:
    * No non-traversed paths
    * No different departure times (expected vs real)
    * No departure before last trip's arrival

    If there i something wrong with the simulation supplied as
    argument, than this function immidiately prints the log of errors.
    Otherwise, if everything checks out, then there will be no prints
    and no output.
    """
    for iteration in range(len(simulation.common_sense)):
        for a in simulation.common_sense[iteration]:
            for p in range(len(a.paths)):
                if a.paths[p] is None:
                    print('Broken paths')
                    print('Iteration', iteration, 'ID', a.id_num, 'Path', p)
                else:
                    for edge in range(len(a.paths[p])-1):
                        if (a.real_edge_times[p][edge][0] is None or 
                            a.real_edge_times[p][edge][-1] is None):
                            print('-----------------------------------------')
                            print('Error in real_edge_times: Iter', iteration) 
                            print('ID', a.id_num, 'Path', p)

                        if (a.expected_edge_times[p][0][0] 
                            != a.real_edge_times[p][0][0]):
                            print('-----------------------------------------')
                            print('Different expected and real edge times:'
                                  + ' Iter', iteration)
                            print('ID', a.id_num, 'Path', p)

            for path_num in range(1,len(a.paths)):
                if a.paths[path_num] is not None:
                    if (a.real_edge_times[path_num][0][0] 
                        < a.real_edge_times[path_num-1][-1][-1] or
                        a.expected_edge_times[path_num][0][0] 
                        < a.expected_edge_times[path_num-1][-1][-1]):
                            print('-----------------------------------------')
                            print('Error in times of departures: Iter', iteration)
                            print('ID', a.id_num, 'Path', path_num)
                    else:
                        pass


def check_FIFO_property_baseline(simulation):
    """The FIFO propert means that any agent cannot traverse 
    edge faster by coming later. 

    Example: if some agent cames to the edge e0 at time t0,
    he will finish traversing it at t2. However, if FIFO property 
    is satisfied any agent who will come to the edge e0 at time 
    (t0 + a) will finish traversing e0 later than t2 for any a.

    This function just checks the consistency of travel time
    function from baseline_dijkstra module for every agent, 
    every time moment and for every edge that was traversed 
    by this agent.

    If there is something wrong, the details for the error
    will be printed, otherwise there will be no prints and
    no output.

    Note, that this function takes a lot of time to run.
    So, use it only when in strong doubt in travel-time
    estimation consistency.
    """
    for iteration in range(len(simulation.common_sense)):
        for a in simulation.common_sense[iteration]:
            for p in range(len(a.paths)):
                for edge in a.edges_passed[p]:
                    check_list = [G[edge[0]][edge[1]]['weight']]
                    for i in range(0, simulation.max_time*100):
                        t = float(i/100)
                        check_list.append(
                            calculate_subjective_weight_for_edge_time(
                                          agent=a,
                                          time=t , 
                                          edge=edge,
                                          iteration=iteration,
                                          statistics=simulation.common_sense))

                        if (t + check_list[-1]/60 
                            <= round(t - 0.01, 2) + check_list[-2]/60):
                            print('FIFO is not satisfied')
                            print('ID',a.id_num,'Path', p, 'Edge', edge, 'Time', t)
                            break


def check_expectations(simulation):
    """Function that compares agents' expectations on the travel
    times on each edge of their path to the least possible time 
    needed to traverse it. So, it checks the consistency of
    expectations.

    If for some agent, his expectations on the travel time appear 
    to be lower than the least possible time to traverse the edge 
    (free-flow time), then the details on the occasion are printed, 
    otherwise there will be no prints.
    """
    index = len(simulation.statistics) - 1
    for a in simulation.statistics[index]:
        for p in range(len(a.paths)):
            for edge_num in range(len(a.paths[p])-1):
                trav_time_exp = a.expected_edge_times[p][edge_num][-1] 
                                - a.expected_edge_times[p][edge_num][0]
                wght = simulation.graph[a.edges_passed[p][edge_num][0]]\
                       [a.edges_passed[p][edge_num][-1]]['weight'] / 60
                
                if round(wght, 2) > round(trav_time_exp, 2):
                    print(simulation)
                    print('ID', a.id_num, 'Path', p, 'Min time', wght, 
                          'Trav exp', round(trav_time_exp, 2))


# Function that compares reality of the travel times on
# each edge of agents' paths to the least possible time needed to traverse it
def check_reality(simulation):
    """Function that compares agents' real travel times
    on each edge of their path to the least possible time 
    needed to traverse it. So, it checks the consistency of
    simulation in terms of path-traversal mechanism.

    If for some agent, his real travel time appear 
    to be lower than the least possible time to traverse the edge 
    (free-flow time), then the details on the occasion are printed, 
    otherwise there will be no prints.
    """
    index = len(simulation.statistics) - 1
    for a in simulation.statistics[index]:
        for p in range(len(a.paths)):
            for edge_num in range(len(a.paths[p])-1):
                trav_time_real = a.real_edge_times[p][edge_num][-1] 
                                 - a.real_edge_times[p][edge_num][0]
                wght = simulation.graph[a.edges_passed[p][edge_num][0]]\
                       [a.edges_passed[p][edge_num][-1]]['weight'] / 60
                
                if round(wght, 2) > round(trav_time_real, 2):
                    print(simulation)
                    print('ID', a.id_num, 'Path', p, 'Min time', wght, 
                          'Trav real', round(trav_time_real, 2))


# FIFO property check
# Takes a lot of time, run only if necessary
def check_FIFO_property_PTA(simulation):
    """The FIFO propert means that any agent cannot traverse 
    edge faster by coming later. 

    Example: if some agent cames to the edge e0 at time t0,
    he will finish traversing it at t2. However, if FIFO property 
    is satisfied any agent who will come to the edge e0 at time 
    (t0 + a) will finish traversing e0 later than t2 for any a.

    This function just checks the consistency of travel time
    function from dijkstra_PTA function for every time moment 
    and for every edge present in the simulation graph. Note, that
    there is no need for agent-wise checks as the travel time
    estimation on some edge and at some time is not related to
    the agent in the PTA simulation.

    If there is something wrong, the details for the error
    will be printed, otherwise there will be no prints and
    no output.

    Note, that this function takes a lot of time to run.
    So, use it only when in strong doubt in travel-time
    estimation consistency.
    """
    for edge in G.edges():
        check_list = [G[edge[0]][edge[1]]['weight']]
        for i in range(0, simulation.max_time):
            t = float(i/100)
            check_list.append(simulation.calculate_weight_for_edge_time_diff(
                                                            arrival_time = t,
                                                            edge = edge))
            if (t + check_list[-1]/60 
                <= round(t - 0.01, 2) + check_list[-2]/60):
                print('FIFO is not satisfied')
                print('ID',a.id_num,'Path', p, 'Edge', edge, 'Time', t)
                break




