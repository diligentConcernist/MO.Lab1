import networkx as nx
import matplotlib.pyplot as plt
from networkx.classes.function import number_of_nodes
import numpy as np
from numpy.lib.function_base import append

class SelfLoopException(Exception):
    pass

class NegativeWeightException(Exception):
    pass

class ConnectivityException(Exception):
    pass

class OutOfRangeException(Exception):
    pass

class EmptyEdgeListException(Exception):
    pass

def readGraph():
    file = open('graph.txt', 'r')
    all = file.read()
    edges = all.splitlines()
    number_of_nodes = edges.pop(0)
    for i in range(len(edges)):
        edges[i] = edges[i].split(", ")
        tmp = []
        for j in range(len(edges[i])):
            tmp.append(int(edges[i][j]))
        edges[i] = tmp    
    file.close()
    return [int(number_of_nodes), edges]       

class Graph:
    def __init__(self, number_of_nodes): 
        self.number_of_nodes = number_of_nodes
        self.edges = []
        self.distance = []
        self.displayed_graph = nx.Graph()
        self.adj_matrix = np.zeros((number_of_nodes, number_of_nodes))
        self.adj_matrix_non_oriented = np.zeros((number_of_nodes, number_of_nodes))

    def addEdge(self, first_node, second_node, width):
        try:
            self.checkRange(first_node)
            self.checkRange(second_node)
        except OutOfRangeException as e:
            print(e)
        else:
            self.edges.append([first_node, second_node, width])
            self.adj_matrix[first_node][second_node] = width
            self.adj_matrix_non_oriented[first_node][second_node] = width
            self.adj_matrix_non_oriented[second_node][first_node] = width

    def checkForLoops(self):
        for i in range(len(self.edges)):
            if (self.edges[i][0] == self.edges[i][1]):
                raise SelfLoopException("Inputed graph contains self loops (node №" + str(i) + ")")

    def checkNegativeWeight(self):
        for i in range(len(self.edges)):
            if (self.edges[i][2] < 0):
                raise NegativeWeightException(
                    "Inputed graph contains edge with negative weight (edge " + 
                    str(self.edges[i][0]) + "-" + str(self.edges[i][1])+ ")")

    def DFS(self, start, visited):
        visited[start] = True
        dfs_visited.append(start)
        for i in range(self.number_of_nodes):
            if (self.adj_matrix_non_oriented[start][i] > 0) and (not visited[i]):
                self.DFS(i, visited)

    def checkForConnectivity(self):
        global dfs_visited
        dfs_visited = []
        visited = [False] * self.number_of_nodes
        self.DFS(1, visited)
        all_nodes = []
        all_nodes.extend(range(0, self.number_of_nodes))
        if not (all_nodes == sorted(dfs_visited)):
            raise ConnectivityException("Inputed graph is not connected")
 
    def checkForEmptiness(self):
        if not self.edges:
            raise EmptyEdgeListException("Inputed graph have no edges")

    def checkRange(self, node_index):
        if (node_index > self.number_of_nodes) or (node_index < 0):
            raise OutOfRangeException("Node index " +  str(node_index) + " is out of range")
            
    def createBasicGraph(self, fig):
        plt.figure(fig)
        self.displayed_graph.clear()
        temp_list = [[(str(elem)) for elem in i] for i in self.edges]
        temp_list = list(map(' '.join, temp_list))
        self.displayed_graph = nx.parse_edgelist(temp_list, nodetype=int, data=(("weight", int),))
        global pos 
        pos = nx.shell_layout(self.displayed_graph)
        nx.draw(self.displayed_graph, pos, with_labels = True)
	
    def Bellman(self, start, finish):
        try:
            self.checkForEmptiness()
            self.checkForLoops()
        except EmptyEdgeListException as e:
            print(e)    
        except SelfLoopException as e:
            print(e)
        else:
            self.distance = [float("Inf")] * self.number_of_nodes
            self.distance[start] = 0

            prev_node = [None] * self.number_of_nodes
            
            for i in range(self.number_of_nodes - 1):
                for first_node, second_node, weight in self.edges:
                    if (self.distance[first_node] != float("Inf") and 
                        self.distance[first_node] + weight < self.distance[second_node]):
                         self.distance[second_node] = self.distance[first_node] + weight
                         prev_node[second_node] = first_node

            try:
                self.checkForConnectivity()
            except ConnectivityException as e:
                print(e)

            else:
                path_weight = self.distance[finish]
            
                if (path_weight == float("Inf")):
                    print("Path from " + str(start) + " to " + str(finish) + " does not exist")

                else:
                    current = finish
                    path = []

                    while (current != None):
                        path.append(current)
                        current = prev_node[current]

                    path_edges = []
                    for i in range(len(path) - 1):
                        path_edges.append([path[i], path[i+1]])

                    path.reverse()
                    print("Optimal path from start to finish is " + str(path) + ", it's length is " + str(int(path_weight)))

                    fig = plt.figure("Bellman")
                    self.createBasicGraph(fig)
                    dist = dict(zip(self.displayed_graph.nodes(), self.distance))
                    nx.set_node_attributes(self.displayed_graph, dist, 'distance')
                
                    labels = nx.get_node_attributes(self.displayed_graph, 'distance')
                    for key, value in pos.items():
                        plt.text(value[0], value[1] + 0.10, s = str(labels[key]), color="red")
            
                    G = nx.DiGraph()
                    G.add_nodes_from(self.displayed_graph)
                    G.add_edges_from(self.displayed_graph.edges)
                    nx.draw(G, pos)
                    labels = nx.get_edge_attributes(self.displayed_graph,'weight')
                    nx.draw_networkx_edge_labels(self.displayed_graph, pos, edge_labels=labels)
                    nx.draw_networkx_edges(self.displayed_graph, pos, edgelist=path_edges, width=3, edge_color="red")

    def MinDistance(self, visited):
        min = float("Inf")
        min_index = -1

        for i in range(self.number_of_nodes):
            if (self.distance[i] < min and visited[i] == False):
                min = self.distance[i]
                min_index = i

        return min_index

    def Dijkstra(self, start, finish):
        self.distance = [float("Inf")] * self.number_of_nodes
        self.distance[start] = 0
        prev_node = [None] * self.number_of_nodes
        visited = [False] * self.number_of_nodes

        for node in range(self.number_of_nodes):
            first_node = self.MinDistance(visited)
            visited[first_node] = True

            for second_node in range(self.number_of_nodes):
                if (self.adj_matrix[first_node][second_node] > 0 and
                    visited[second_node] == False and
                    self.distance[second_node] > self.distance[first_node] + self.adj_matrix[first_node, second_node]):
                        self.distance[second_node] = self.distance[first_node] + self.adj_matrix[first_node, second_node]
                        prev_node[second_node] = first_node

        try:
            self.checkForConnectivity()
        except ConnectivityException as e:
            print(e)

        else:
            path_weight = self.distance[finish]
            
            if (path_weight == float("Inf")):
                print("Path from " + str(start) + " to " + str(finish) + " does not exist")

            else:
                current = finish
                path = []
                while (current != None):
                    path.append(current)
                    current = prev_node[current]
                path_edges = []
                for i in range(len(path) - 1):
                    path_edges.append([path[i], path[i+1]])

                path.reverse()
                print("Optimal path from start to finish is " + str(path) + ", it's length is " + str(int(path_weight)))
                fig = plt.figure("Dijkstra")
                self.createBasicGraph(fig)

                self.distance = dict(zip(self.displayed_graph.nodes(), self.distance))
                nx.set_node_attributes(self.displayed_graph, self.distance, 'distance')
                labels = nx.get_node_attributes(self.displayed_graph, 'distance')
                for key, value in pos.items():
                    plt.text(value[0], value[1] + 0.10, s = str(labels[key]), color="red")
            
                G = nx.DiGraph()
                G.add_nodes_from(self.displayed_graph)
                G.add_edges_from(self.displayed_graph.edges)
                nx.draw(G, pos)
                labels = nx.get_edge_attributes(self.displayed_graph,'weight')
                nx.draw_networkx_edge_labels(self.displayed_graph, pos, edge_labels=labels)
                nx.draw_networkx_edges(self.displayed_graph, pos, edgelist=path_edges, width=3, edge_color="red")

    def showResult(self):
        plt.show()    
        
def createGraph():
    tmp = readGraph()
    number_of_nodes = tmp[0]
    edge_list = tmp[1]
    G = Graph(number_of_nodes)
    for i in range(len(edge_list)):
        G.addEdge(edge_list[i][0], edge_list[i][1], edge_list[i][2])
    return G

H = createGraph()
print(H.edges)
H.Bellman(0, 4)
H.Dijkstra(0, 4)
H.showResult()
