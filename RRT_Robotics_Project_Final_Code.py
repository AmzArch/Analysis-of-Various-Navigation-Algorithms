import random
import math
import pygame


import numpy as np
from heapdict import heapdict
from math import floor, sqrt

import itertools
from multiprocessing import Pool
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import colors
import networkx as nx

import time
from statistics import mean

# TYPE_MAPPING is used to colour the tile according to the type
# e.g. all obstacles are in black colour. these tiles will be inaccessible
TYPE_MAPPING = {
    'obstacle' : 'black',
    'source' : 'royalblue',
    'target' : 'green',
    'path' : 'peachpuff',
    'others' : 'white',
    'visited' : 'yellow'
}

# COLOUR_MAPPING is used to assign a value in numpy array so that matplotlib knows which value will be mapped to which colour
# e.g. if coordinate (1,2) has value 0, the tile will be of black colour
COLOUR_MAPPING = {
    'black' : 0,
    'royalblue' : 1,
    'green' : 2,
    'peachpuff'  : 3,
    'white' : 4,
    'yellow' : 5
}

# The common map creation class used for all methods with seed=0
# The definitions of these functions can be found in the colab notebook
class Graph:

    def __init__(self, w, h, seed=0):
        self.COLOUR_MAPPING = COLOUR_MAPPING
        self.TYPE_MAPPING = TYPE_MAPPING

        self.cmap = colors.ListedColormap( list( self.COLOUR_MAPPING.keys() ) )
        self.bounds= list( self.COLOUR_MAPPING.values() ) + [ max(self.COLOUR_MAPPING.values()) +1 ]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

        self.w = w
        self.h = h
        self.g = np.full(
            (h, w),
            self.COLOUR_MAPPING[self.TYPE_MAPPING['others']],
            dtype=np.uint8
        )
        self.seed = seed

        self.adjacency_list = {}
        self.source = None
        self.target = None
        self.obstacles = []

    def add_obstacle(self, x, y):
        self.g[y,x] = self.COLOUR_MAPPING[ self.TYPE_MAPPING['obstacle'] ]
        self.obstacles += [(x,y)]

    def add_source(self, x, y):
        # reset previous source to ensure that there is only one source
        if self.source is not None:
            self.g[self.source[1],self.source[0]] = self.COLOUR_MAPPING[ self.TYPE_MAPPING['others'] ]

        # update new source
        self.g[y,x] = self.COLOUR_MAPPING[ self.TYPE_MAPPING['source'] ]
        self.source = (x,y)

    def add_target(self, x, y):
        # reset previous target to ensure that there is only one target
        if self.target is not None:
            self.g[self.target[1],self.target[0]] = self.COLOUR_MAPPING[ self.TYPE_MAPPING['others'] ]

        # update new target
        self.g[y,x] = self.COLOUR_MAPPING[ self.TYPE_MAPPING['target'] ]
        self.target = (x,y)

    def generate_random_graph(self, probability_obstacle=0.2):
        np.random.seed(self.seed)

        # generate list of all nodes
        nodes = [ (x,y) for x in range(self.w) for y in range(self.h) ]
        nodes_idx = list( range(self.w * self.h) )
        # print(f"nodes : {nodes}")

        # pick one node as the source and remove node from the list
        source_idx = np.random.choice(nodes_idx)
        nodes_idx.remove(source_idx)
        # print(f"source : {source_idx}\n nodes : {nodes_idx}")
        self.add_source(nodes[source_idx][0], nodes[source_idx][1])

        # pick one node as the target and remove node from the list
        target_idx = np.random.choice(nodes_idx)
        nodes_idx.remove(target_idx)
        # print(f"target : {target_idx}\n nodes : {nodes_idx}")
        self.add_target(nodes[target_idx][0], nodes[target_idx][1])

        # pick a few nodes as obstacles
        number_obstacles = floor(probability_obstacle * self.w * self.h)
        obstacles_idx = np.random.choice(nodes_idx, size=number_obstacles, replace=False)
        # print(f"obstacles : {obstacles_idx}")
        for obstacles_idx in obstacles_idx:
            self.add_obstacle(nodes[obstacles_idx][0], nodes[obstacles_idx][1])

    def add_path(self, x, y):
        # only add path if target or source is not (x,y)
        if (x,y) != self.target and (x,y) != self.source:
            self.g[y,x] = self.COLOUR_MAPPING[ self.TYPE_MAPPING['path'] ]

    def add_paths(self, paths):
        for x,y in paths:
            self.add_path(x,y)

    # Add a single visited node to the list of visited nodes.
    def add_visited_node(self, x, y):
        if (x,y) != self.target and (x,y) != self.source:
            self.g[y,x] = self.COLOUR_MAPPING[ self.TYPE_MAPPING['visited'] ]

    # Add a list of visited nodes to the list of visited nodes.
    def add_visited(self, paths):
        for x,y in paths:
            self.add_visited_node(x,y)

    def plot(self, plot_type, distance_fn=None):
        if plot_type == 'grid':
            plt.figure( figsize=(self.w, self.h) )

            plt.imshow(
                self.g,
                interpolation = 'none',
                cmap = self.cmap,
                norm = self.norm,
                extent=[0, self.w, self.h, 0] # to ensure that the coordinate plot is adjusted
            )
            plt.ylim((0, self.h))
            plt.xlim((0, self.w))
            plt.xticks(np.arange(0., self.w + 1., 1.))
            plt.yticks(np.arange(0., self.h + 1., 1.))
            plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
            plt.show()

        elif plot_type == 'grid_pygame':
            plt.figure( figsize=(self.w, self.h) )

            plt.imshow(
                self.g,
                interpolation = 'none',
                cmap = self.cmap,
                norm = self.norm
            )
            plt.ylim((0, self.h))
            plt.xlim((0, self.w))
            plt.xticks(np.arange(0., self.w + 1., 1.))
            plt.yticks(np.arange(0., self.h + 1., 1.))
            plt.show()

        elif plot_type == 'graph' and distance_fn is not None:
            # source : https://stackoverflow.com/questions/28372127/add-edge-weights-to-plot-output-in-networkx
            plt.figure( figsize=(self.w, self.h) )

            self.g_nx = nx.Graph()
            self.adjacency_list = self.get_adjacency_list(distance_fn)

            nodes = [ (x,y) for x in range(self.w) for y in range(self.h) ]

            for node in nodes:
                self.g_nx.add_node(node, pos=node)

            for node in self.adjacency_list.keys():
                for neighbour, distance in self.adjacency_list[node].items():
                    self.g_nx.add_edge(node, neighbour, weight=distance)

            pos = nx.get_node_attributes(self.g_nx, 'pos')
            nx.draw(self.g_nx, pos)
            labels = nx.get_edge_attributes(self.g_nx, 'weight')
            nx.draw_networkx_edge_labels(self.g_nx, pos, edge_labels=labels)


    def is_valid_coordinate(self, x, y):
        return (x >= 0) and (x < self.w) and (y >= 0) and (y < self.h)

    def add_neighbour(self, current, neighbour, distance_fn):
        x, y = neighbour
        if self.is_valid_coordinate(x, y):
            if self.g[y, x] != self.COLOUR_MAPPING[ self.TYPE_MAPPING['obstacle'] ]:
                self.adjacency_list[current][neighbour] = distance_fn(current, neighbour)

    def get_adjacency_list(self, distance_fn):
        '''
        distance_fn needs to be a function that expects 2 inputs and returns float / int
        for constant distance : lambda current, neighbour : 3
        for linear distance : lambda current, neighbour : abs(current[0] - neighbour[0]) + abs(current[1] - neighbour[1])
        for random distance : lambda current, neighbour : np.random.randint(10)

        return adjacency list with the following format
        { (x_source, y_source) :
            {
                (x_neighbour, y_neighbour) : distance_to_neighbour,
                (x_neighbour, y_neighbour) : distance_to_neighbour
          }
        }
        '''

        self.adjacency_list = {}
        np.random.seed(self.seed)
        for x in range(self.w):
            for y in range(self.h):
                # obstacle node is not added in the adjacency list
                if self.g[y,x] != self.COLOUR_MAPPING[ self.TYPE_MAPPING['obstacle'] ]:
                    self.adjacency_list[ (x,y) ] = {}

                    self.add_neighbour((x,y), (x-1, y-1), distance_fn) # add bottom left
                    self.add_neighbour((x,y), (x, y-1), distance_fn) # add bottom
                    self.add_neighbour((x,y), (x+1, y-1), distance_fn) # add bottom right
                    self.add_neighbour((x,y), (x+1, y), distance_fn) # add right
                    self.add_neighbour((x,y), (x+1, y+1), distance_fn) # add top right
                    self.add_neighbour((x,y), (x, y+1), distance_fn) # add top
                    self.add_neighbour((x,y), (x-1, y+1), distance_fn) # add top left
                    self.add_neighbour((x,y), (x-1, y), distance_fn) # add left
        return self.adjacency_list




# This class is used to create the PyGame surface map from the GridWorld Map
class RRTMap:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
        self.start = start
        self.goal = goal
        self.MapDimensions = MapDimensions
        self.Maph, self.Mapw = self.MapDimensions

        # window settings
        self.MapWindowName = 'RRT path planning'
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.Mapw, self.Maph))
        self.map.fill((255, 255, 255))
        self.nodeRad = 2
        self.nodeThickness = 0
        self.edgeThickness = 1
        self.obstacles = []
        self.obsdim = obsdim
        self.obsNumber = obsnum

        # Colors
        self.grey = (70, 70, 70)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.white = (255, 255, 255)

    # Draw map
    def drawMap(self, obstacles):
        pygame.draw.circle(self.map, self.Green, self.start, self.nodeRad + 5, 0)
        pygame.draw.circle(self.map, self.Green, self.goal, self.nodeRad + 20, 1)
        self.drawObs(obstacles)

    #This function, highlights the nodes in the final path and makes them red dots
    def drawPath(self, path):
        for node in path:
            pygame.draw.circle(self.map, self.Red, node, 3, 0)


    # This function is used to draw obstacle from the obstacle list retrieved from the grid GridWorld
    def drawObs(self, obstacles):
        obstaclesList = obstacles.copy()
        while (len(obstaclesList) > 0):
            obstacle = obstaclesList.pop(0)
            pygame.draw.rect(self.map, self.grey, obstacle)


class RRTGraph:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.maph, self.mapw = MapDimensions
        self.x = []
        self.y = []
        self.parent = []
        # initialize the tree
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)
        # the obstacles
        self.obstacles = []
        self.obsDim = obsdim
        self.obsNum = obsnum
        # path
        self.goalstate = None
        self.path = []

    # Add a node at nth postion, with corrdinates x and y
    def add_node(self, n, x, y):
        self.x.insert(n, x)
        self.y.append(y)

    # Remove the nth node
    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)

    # add and create an edge between the the child and parent nodes as specified
    def add_edge(self, parent, child):
        self.parent.insert(child, parent)

    # Remove the nth edge
    def remove_edge(self, n):
        self.parent.pop(n)

    # Get the number of nodes on the map
    def number_of_nodes(self):
        return len(self.x)

    # get distance between node n1 and n2
    def distance(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        px = (float(x1) - float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        return (px + py) ** (0.5)

    # Choose a random point on the map
    def sample_envir(self):
        x = int(random.uniform(0, self.mapw))
        y = int(random.uniform(0, self.maph))
        return x, y

    # Get the nearest node to the specified nth node
    def nearest(self, n):
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.distance(i, n) < dmin:
                dmin = self.distance(i, n)
                nnear = i
        return nnear

    # Check if the last created node occupies free space or not and remove if colliding with obstacle
    def isFree(self):
        n = self.number_of_nodes() - 1
        (x, y) = (self.x[n], self.y[n])
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectang = obs.pop(0)
            if rectang.collidepoint(x, y):
                self.remove_node(n)
                return False
        return True

    # Check if lines between two points crosses an obstacle
    def crossObstacle(self, x1, x2, y1, y2):
        obs = self.obstacles.copy()
        while (len(obs) > 0):
            rectang = obs.pop(0)
            for i in range(0, 101):
                u = i / 100
                x = x1 * u + x2 * (1 - u)
                y = y1 * u + y2 * (1 - u)
                if rectang.collidepoint(x, y):
                    return True
        return False

    # Connect two nodes
    def connect(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        if self.crossObstacle(x1, x2, y1, y2):
            self.remove_node(n2)
            return False
        else:
            self.add_edge(n1, n2)
            return True

    # Create a node and edge and node towards the random node within max step defined (10 for us)
    def step(self, nnear, nrand, dmax=10):
        d = self.distance(nnear, nrand)
        if d > dmax:
            u = dmax / d
            (xnear, ynear) = (self.x[nnear], self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py, px)
            (x, y) = (int(xnear + dmax * math.cos(theta)),
                      int(ynear + dmax * math.sin(theta)))
            self.remove_node(nrand)
            if abs(x - self.goal[0]) <= dmax and abs(y - self.goal[1]) <= dmax:
                self.add_node(nrand, self.goal[0], self.goal[1])
                self.goalstate = nrand
                # print(f"Goalstate={self.goalstate}")
                self.goalFlag = True
            else:
                self.add_node(nrand, x, y)

    #Explore map towards the goal
    def bias(self, ngoal):
        n = self.number_of_nodes()
        self.add_node(n, ngoal[0], ngoal[1])
        nnear = self.nearest(n)
        self.step(nnear, n)
        self.connect(nnear, n)
        return self.x, self.y, self.parent

    # Explore map in entirety
    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_envir()
        self.add_node(n, x, y)
        if self.isFree():
            xnearest = self.nearest(n)
            self.step(xnearest, n)
            self.connect(xnearest, n)
        return self.x, self.y, self.parent

    #Check if path has reached goal and return the path
    def path_to_goal(self):
        if self.goalFlag:
            self.path = []
            self.path.append(self.goalstate)
            # print(f"path={self.path}")
            # print(f"goalstate={self.goalstate}")
            newpos = self.parent[self.goalstate]
            # print(f"newpos={newpos}")
            while (newpos != 0):
                self.path.append(newpos)
                newpos = self.parent[newpos]
            self.path.append(0)
        return self.goalFlag

    # Get Coordinates of the path
    def getPathCoords(self):
        pathCoords = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords.append((x, y))
        return pathCoords

    # Get the cost of the path
    def cost(self, n):
        ninit = 0
        n = n
        parent = self.parent[n]
        c = 0
        while n is not ninit:
            c = c + self.distance(n, parent)
            n = parent
            if n is not ninit:
                parent = self.parent[n]
        return c

    # This is an extra and is used to work for real life scenarios where obstacle are expanded
    # Obstacles are expanded so we take into consideration the robots size
    def getTrueObs(self, obs):
        TOBS = []
        for ob in obs:
            TOBS.append(ob.inflate(-50, -50))
        return TOBS


    # Check if obstacle do not collide with start and end point (Was used in random obstacle generation before) and then append it to final obstacle list
    def makeobs(self, obslist, sc, Y):
        obs = []
        for i in range(0, self.obsNum):
            rectang = None
            startgoalcol = True
            while startgoalcol:
                upperx, uppery = obslist[i]
                upper = (upperx*sc, sc*(Y-uppery-1))
                # print(f"upper={upper}")
                rectang = pygame.Rect(upper, (self.obsDim, self.obsDim))
                if rectang.collidepoint(self.start) or rectang.collidepoint(self.goal):
                    startgoalcol = True
                else:
                    startgoalcol = False
                obs.append(rectang)
            self.obstacles = obs.copy()
        return obs




def RRT_Run(x,y,p, file):
    # generate graph with width = x and height = y, density of obstacle = p
    # seed can be set to ensure that we can attain the same graph again after every rerun
    # Default seed = 0 is used
    g = Graph(x, y)

    g.generate_random_graph(probability_obstacle=p)

    # sc is the scale that is used that is one node in griwordl is sc*sc pixels in the RRT PyGame Surface
    sc = 20
    #Get dimensions of pixelated surface
    dimensions = (sc*g.w+1,sc*g.h+1)
    startx, starty = g.source
    # Scale Everything to Pixelated Surface and invert y axis at the same time as the axis of the gridworld start at bottom left while PyGame surface starts at Top Left
    start=(sc*(startx+0.5),sc*(g.h-starty-0.5))
    goalx, goaly = g.target
    goal=(sc*(goalx+0.5),sc*(g.h-goaly-0.5))
    obsdim=sc
    obsnum=len(g.obstacles)
    obslist = g.obstacles
    # print(f"Obslist={obslist} Obsnum={obsnum}")
    iteration=0
    t1=0

    pygame.init()
    map=RRTMap(start,goal,dimensions,obsdim,obsnum)
    graph=RRTGraph(start,goal,dimensions,obsdim,obsnum)

    obstacles=graph.makeobs(obslist, sc, g.h)
    map.drawMap(obstacles)

    t1=time.time()
    #Run the RRT_Run function till we get a path
    while (not graph.path_to_goal()):
        time.sleep(0.005)
        elapsed=time.time()-t1
        t1=time.time()
        #raise exception if timeout
        if elapsed > 10:
            print('timeout re-initiating the calculations')
            raise
        # Introduce Bias to explore once in 10 times towards the goal, all other 9 times, will explore randomly and outward
        if iteration % 10 == 0:
            X, Y, Parent = graph.bias(goal)
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad*2, 0)
            pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]),
                             map.edgeThickness)

        else:
            X, Y, Parent = graph.expand()
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad*2, 0)
            pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]),
                             map.edgeThickness)

        if iteration % 5 == 0:
            pygame.display.update()
        iteration += 1
    map.drawPath(graph.getPathCoords())
    # print(f"The cost is {graph.cost(n = graph.number_of_nodes())}")
    # print("Cost is " + str((graph.cost(graph.number_of_nodes()-1))/sc))
    # print("Total nodes Visited is " + str(graph.number_of_nodes()))
    # print("Path is " + str(len(graph.path)))

    pygame.display.update()
    pygame.event.clear()
    # pygame.event.wait(0)
    pygame.image.save(map.map, str(file))
    return(graph.cost(graph.number_of_nodes()-1)/sc, graph.number_of_nodes(), graph.path)


# This is used to collect the avergae of 50 runs on each map
def av_RRT_runs(xm, ym, pm, its):
    cost_l = []
    nodes_l = []
    path_l = []

    for i in range(its):
        result = False
        while not result:
            try:
                cost, nodes, path = RRT_Run(x = xm, y = ym, p = pm, file = f"{xm}_{ym}_obs_{pm}_path{i}.png")
                result = True
            except:
                result = False
        cost_l.append(cost)
        nodes_l.append(nodes)
        path_l.append(len(path))

    print("Average Cost is "+ str(mean(cost_l)))
    print("Average number of Nodes visited is "+ str(mean(nodes_l)))
    print("Average Path Lenght is "+ str(mean(path_l)))
    df = pd.DataFrame({
        'Cost': cost_l,
        'Nodes Visited': nodes_l,
        'Path Lenght': path_l
     })

    df.to_csv(f"{xm}_{ym}_obs_{pm}.csv")


# The variable are the X dimension of the graph, Y dimension, Obstacle Density, and Number of runs we average the results for
# To change the scale we use to convert tiles to pixels with change the variable sc on line 505
av_RRT_runs(20,20,0.2,10)
