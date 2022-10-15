#mixhop_generator.py

import pickle
import networkx as nx
import numpy as np
random_state = np.random.RandomState(0) # FIX THIS

def random_split_counts(total_number, no_of_splits):
    split_indices = np.sort(np.random.choice(total_number,no_of_splits-1, replace = False)+1)
    split_indices = np.insert(split_indices, 0, 0)
    split_indices = np.append(split_indices, total_number)
    split_counts = np.diff(split_indices)
    return split_counts

class GraphGenerator:
	def __init__(self, numClass):
		self.numClass = numClass

	def format_path(self, G, savePath, graphName, **kwargs):
		graphName = graphName.format(numNode=G.number_of_nodes(), numEdge=G.number_of_edges(), numClass=self.numClass, **kwargs)
		savePath = savePath.format(graphName=graphName, numNode=G.number_of_nodes(), numEdge=G.number_of_edges(), numClass=self.numClass, **kwargs)
		return savePath, graphName

	def save_graph(self, G: nx.Graph, savePath, graphName, **kwargs):
		savePath, graphName = self.format_path(G, savePath, graphName, **kwargs)
		G_d = nx.to_dict_of_lists(G)
		print("Saving graph to {}".format(os.path.join(savePath, graphName + ".graph")))
		pickle.dump(G_d, open(os.path.join(savePath, graphName + ".graph"), "wb"))
	
	def save_y(self, G: nx.Graph, savePath, graphName, **kwargs):
		savePath, graphName = self.format_path(G, savePath, graphName, **kwargs)
		ally = np.zeros((len(G.nodes()), self.numClass))
		for v in G.nodes():
			ally[v][G.nodes[v]['color'] - 1] = 1
		print("Saving labels to {}".format(os.path.join(savePath, graphName + ".ally")))
		pickle.dump(ally, open(os.path.join(savePath, graphName + ".ally"), "wb"))
	
	def save_nx_graph(self, G: nx.Graph, savePath, graphName, **kwargs):
		savePath, graphName = self.format_path(G, savePath, graphName, **kwargs)
		print("Pickling networkx graph to {}".format(os.path.join(savePath, graphName + ".gpickle.gz")))
		nx.write_gpickle(G, os.path.join(savePath, graphName + ".gpickle.gz"))

class MixhopGraphGenerator(GraphGenerator):
	def get_color(self, class_ratio): # Assign new node to a class
		if self.__coloriter:
			return next(self.__coloriter)
		else:
			return np.random.choice(list(range(1, len(class_ratio) + 1)), 1, False, class_ratio)[0]

	def color_weight(self, col1, col2):
		dist = abs(col1 - col2)
		dist = min(dist, len(self.classRatio) - dist)
		return self.heteroWeightsDict[dist]
	
	def get_neighbors(self, G, m, col, h):
		pr = dict()
		for v in G.nodes():
			degree_v = float(max(G.degree[v], 1)) # Degree is treated as at least 1, so that pr[:] is not all 0
			if G.nodes[v]['color'] == col:
				pr[v] = float(degree_v) * h
			else:
				pr[v] = float(degree_v) * ((1 - h) * self.color_weight(col, G.nodes[v]['color']))

		norm_pr = float(sum(pr.values()))
		if norm_pr == 0:
			return None
		else:
			for v in list(pr.keys()):
				pr[v] = float(pr[v]) / norm_pr
			neighbors = np.random.choice(list(pr.keys()), m, False, list(pr.values()))
			return neighbors

	def __init__(self, classRatio, heteroClsWeight="circularDist", **kwargs):
		super().__init__(len(classRatio))
		self.classRatio = classRatio
		self.heteroWeightsDict = dict()
		
		if heteroClsWeight == "circularDist":
			for i in range(2, self.numClass + 1):
				circularDist = min(i - 1, self.numClass - (i - 1))
				self.heteroWeightsDict[circularDist] = self.heteroWeightsDict.get(circularDist, 0) + 1

			maxDist = max(self.heteroWeightsDict.keys())
			weightSum = 0
			for dist, times in self.heteroWeightsDict.items():
				self.heteroWeightsDict[dist] = kwargs["heteroWeightsExponent"] ** (maxDist - dist)
				weightSum += self.heteroWeightsDict[dist] * times
			self.heteroWeightsDict = {dist: weight / weightSum for dist, weight in self.heteroWeightsDict.items()}
		
		elif heteroClsWeight == "uniform":
			for i in range(2, self.numClass + 1):
				circularDist = min(i - 1, self.numClass - (i - 1))
				self.heteroWeightsDict[circularDist] = 1.0 / len(range(2, self.numClass + 1))

	def generate_graph(self, n, m, m0, h):
		'''
		n: Target size for the generated network
		m: number of edges added with each new node
		m0: number of nodes to begin with
		h: homophily
		'''
		if n > 1 and np.sum(self.classRatio) == n:
			#print("Graph will be generated with size of each class exactly equal to the number specified in classRatio.")
			self.__colorlist = []
			for classID, classSize in enumerate(self.classRatio):
				self.__colorlist += [classID + 1] * int(classSize - m)
			random_state.shuffle(self.__colorlist)
			head_list = list(range(1, self.numClass + 1)) * m
			random_state.shuffle(head_list)
			self.__colorlist = head_list + self.__colorlist
			self.__coloriter = iter(self.__colorlist)
		else:
			self.__coloriter = None
		
		if m * self.numClass > m0:
			raise ValueError("Barabasi-Albert model requires m to be less or equal to m0")

		if m > n:
			raise ValueError("m > n should be satisfied")

		G = nx.Graph()

		for v in range(m0):
			next_color = self.get_color(self.classRatio)
			if v > 1:
				if h != 0 and h != 1:
					next_neighbor = v - 1
				else:
					next_n = self.get_neighbors(G, 1, next_color, h)
					if next_n is not None:
						next_neighbor = next_n[0]
					else:
						next_neighbor = None

			G.add_node(v, color=next_color)
			if v > 1 and next_neighbor is not None:
				G.add_edge(v, next_neighbor)
			
		for v in range(m0, n):
			if v % 1000 == 0:
				print("Generating graph... Now processing v = {}".format(v))
			col = self.get_color(self.classRatio)
			us = self.get_neighbors(G, m, col, h)

			G.add_node(v, color=col)
			assert us is not None
			for u in us:
				G.add_edge(v, u)

		assert len(list(nx.selfloop_edges(G))) == 0
		return G

	def generate_graph_contaminated(self, n, m, m0, h, contamination = 1.0):
		'''
		n: Target size for the generated network
		m: number of edges added with each new node
		m0: number of nodes to begin with
		h: homophily
		'''
		if n > 1 and np.sum(self.classRatio) == n:
			#print("Graph will be generated with size of each class exactly equal to the number specified in classRatio.")
			self.__colorlist = []
			for classID, classSize in enumerate(self.classRatio):
				self.__colorlist += [classID + 1] * int(classSize - m)
			random_state.shuffle(self.__colorlist)
			head_list = list(range(1, self.numClass + 1)) * m
			random_state.shuffle(head_list)
			self.__colorlist = head_list + self.__colorlist
			self.__coloriter = iter(self.__colorlist)
		else:
			self.__coloriter = None
		
		if m * self.numClass > m0:
			raise ValueError("Barabasi-Albert model requires m to be less or equal to m0")

		if m > n:
			raise ValueError("m > n should be satisfied")

		G = nx.Graph()

		for v in range(m0):
			next_color = self.get_color(self.classRatio)
			if v > 1:
				if h != 0 and h != 1:
					next_neighbor = v - 1
				else:
					r = np.random.uniform()
					if r < contamination/2:
						changed_h = h + 0.4
					elif r < contamination:
						changed_h = h - 0.4
					else:
						changed_h = h
					next_n = self.get_neighbors(G, 1, next_color, changed_h)
					if next_n is not None:
						next_neighbor = next_n[0]
					else:
						next_neighbor = None

			G.add_node(v, color=next_color)
			if v > 1 and next_neighbor is not None:
				G.add_edge(v, next_neighbor)
			
		for v in range(m0, n):
			if v % 1000 == 0:
				print("Generating graph... Now processing v = {}".format(v))
			col = self.get_color(self.classRatio)

			r = np.random.uniform()
			if r < contamination/2:
				changed_h = h + 0.4
			elif r < contamination:
				changed_h = h - 0.4
			else:
				changed_h = h
			us = self.get_neighbors(G, m, col, changed_h)

			G.add_node(v, color=col)
			assert us is not None
			for u in us:
				G.add_edge(v, u)

		assert len(list(nx.selfloop_edges(G))) == 0
		return G

	def __call__(self, n, m, m0, h):
		return self.generate_graph(n, m, m0, h)

	def save_graph(self, G:nx.Graph, savePath="{graphName}", graphName="{method}-n{numNode}-h{h}-c{numClass}", **kwargs):
		super().save_graph(G, savePath, graphName, method="mixhop", **kwargs)

	def save_y(self, G:nx.Graph, savePath="{graphName}", graphName="{method}-n{numNode}-h{h}-c{numClass}", **kwargs):
		super().save_y(G, savePath, graphName, method="mixhop", **kwargs)