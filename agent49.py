"""
This file implements the Agent class.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar,Iterator
from queue import Queue, LifoQueue, PriorityQueue
from Maze import Action, Maze
from util import Coord, SearchResult
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt



class Node:
    """
    First, we define the Node class, that imitates the nodes mentioned in lecture. 
    A node represents a step along the way of searching for the goal state in a search tree/graph.
    Remember - a node is not the same as a coordinate. It contains more information.
    """

    def __init__(self, coord: Coord, parent:'Node' | None = None,
                 action: Action | None = None, path_cost=0):
        self.coord = coord
        self.parent = parent
        self.action = action
        self.path_cost = path_cost


    # Given a node, get a path back to the start node by recording the actions taken from the parent node
    def trace_back(self) -> tuple[list[Coord], list[Action]]:
        coord_path = []
        actions = []
        trace: Node = self
        while trace is not None:
            coord_path.append(trace.coord)
            if trace.action is not None:
                actions.append(trace.action)
            trace = trace.parent
        coord_path.reverse()
        actions.reverse()
        return coord_path, actions


    """
    Algorithms like UCS require us to compare and order nodes.
    Since nodes are objects, we can define a custom comparator via operator overriding.
    The three functions below override the standard "==", "<" and ">" operators for the node class.
    This allows us to implement logic like node1 > node2.
    More importantly, these operators are used by the queue data structures we imported at the start.
    """
    def __eq__(self, other):
        return self.path_cost == other.path_cost


    def __lt__(self, other):
        return self.path_cost < other.path_cost


    def __gt__(self, other):
        return self.path_cost > other.path_cost
    
    
    """
    The ability to hash a node allows you to use them as members of a `set` or
    as a key in a `dict`.
    """
    def __hash__(self):
        return hash((self.coord,
                     self.parent.coord if self.parent is not None else None,
                     self.action if self.action is not None else None,
                     self.path_cost))


"""
Next we define the AStar node. Note that the AStar node is identical to a regular node but has two key differences
1. The AStar node has an end coordinate
2. The AStar node has a heuristic value, computed using a heuristic function
"""
class AStarNode(Node):
    """
    HINT: Make sure this is set first before using the heuristics function
    Remember that you need to set this for the entire class, not just one object
    """
    END_COORDS = ClassVar[set[Coord]]

    def __init__(self, coord: Coord, parent: Node | None = None,
                    action: Action | None = None, cost: int = 0):
        super().__init__(coord, parent, action, cost)
        # store the heuristics value for this node
        self.h_val = self.heuristic_function()


    """
    DONE - complete the heuristic function
    Implement the heuristic function for the AStar Node by returning the minimum euclidean (straight line) distance
    between the AStar Node's coordinate and the END_COORDS。
    Hint: You may use numpy's sqrt function, and the builtin min function
    """
    def heuristic_function(self):
        x = self.coord[0] 
        y = self.coord[1]

        min_dist = min(
            np.sqrt((end[0] - x) ** 2 + (end[1] - y) ** 2)
            for end in AStarNode.END_COORDS
        )

        return min_dist

    """
    DONE - complete the operator overloads for the AStar node class
    Unlike the regular node, AStar nodes are not compared using just the path costs。
    Complete the operator overloads for the AStar nodes.
    Hint: You can use the same syntax as the overload in the base node class。
    """
    def __eq__(self, other):
        return (
            self
            .path_cost + self
            .heuristic_function() == other
            .path_cost + other
            .heuristic_function()
        )

    def __lt__(self, other):
        return (
            self
            .path_cost + self
            .heuristic_function() < other
            .path_cost + other
            .heuristic_function()
        )


    def __gt__(self, other):
        return (
            self
            .path_cost + self
            .heuristic_function() > other
            .path_cost + other
            .heuristic_function()
        )
    
    """
    The ability to hash a node allows you to use them as members of a `set` or
    as a key in a `dict`.
    """
    def __hash__(self):
        return hash((self.coord,
                     self.parent.coord if self.parent is not None else None,
                     self.action if self.action is not None else None,
                     self.path_cost,
                     self.h_val,
                     self.END_COORDS))


@dataclass
class Agent:
    """This is the Agent class. This class mimics an agent that explores the
    maze. The agent has just two attributes - the maze that it is in, and the
    node expansion history. The expansion history is used just to evaluate your
    implementation.
    """
    maze: Maze
    expansion_history: list[Coord] = field(default_factory=list)

    # We want to reset this every time we use a new algorithm
    def clear_expansion_history(self):
        self.expansion_history.clear()


    # Visualize the maze and how it was explored in matplotlib
    def visualize_expansion(self, path):
        plt.subplot(1, 1, 1)
        blocks = np.zeros((self.maze.height, self.maze.width))
        blocks[:] = np.nan
        for co_ord in self.maze.pits:
            blocks[co_ord[1], co_ord[0]] = 2

        expansion_cval = np.zeros((self.maze.height, self.maze.width))

        for i, coord in enumerate(self.expansion_history):
            expansion_cval[coord[1], coord[0]] = len(self.expansion_history) - i + len(self.expansion_history)

        plt.pcolormesh(
            expansion_cval,
            shading='flat',
            edgecolors='k', linewidths=1, cmap='Blues')

        cmap = colors.ListedColormap(['grey', 'grey'])

        plt.pcolormesh(
            blocks,
            shading='flat',
            edgecolors='k', linewidths=1, cmap=cmap)

        start = self.maze.start
        ends = self.maze.end

        # Plot start and end points
        plt.scatter(start[0] + 0.5, start[1] + 0.5, color='red', s=100, marker='o', label='Start')
        for end in ends:
            plt.scatter(end[0] + 0.5, end[1] + 0.5, color='gold', s=100, marker=(5, 1), label='End')

        plt.title("Maze Plot")
        plt.xlabel("X")
        plt.ylabel("Y", rotation=0)

        plt.xticks(np.arange(0 + 0.5, expansion_cval.shape[1] + 0.5), np.arange(0, expansion_cval.shape[1]))
        plt.yticks(np.arange(0 + 0.5, expansion_cval.shape[0] + 0.5), np.arange(0, expansion_cval.shape[0]))

        # Plot the path only if it exists
        if path is not None:
            for i in range(len(path) - 1):
                x, y = path[i]
                next_x, next_y = path[i + 1]
                plt.annotate('', xy=(next_x + 0.5, next_y + 0.5), xytext=(x + 0.5, y + 0.5),
                             arrowprops=dict(color='g', arrowstyle='->', lw=2))
        plt.show()


    """
    Done: Complete the goal_test function
    Input: a node object
    Returns: A boolean indicating whether or not the node corresponds to a goal
    Hint: The agent has the maze object as an attribute. The maze object has a set of end coordinates
    You can use the 'in' operator to determine if an object is in a set
    """
    def goal_test(self, node: Node) -> bool:
        # Your implementation here :)
        return node.coord in self.maze.end
    

    """
    Done: Complete the expand_node function
    For each neighbouring node that can be reached from the current node within one action, 'yield' the node.
    Hints:
    1.  Use self.maze.valid_ordered_action(...) to obtain all valid action for a given coordinate
    2.  Use Maze.resulting_coord(...) to compute the new states. Note the capital M is important, since it's a
        class function
    3.  Use yield to temporarily return a node to the caller function while saving the context of this function.
        When expand is called again, execution will be resumed from where it stopped.
        Follow this link to learn more:
        https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    4.  Take advantage of polymorphic construction. You need to ensure that a Node object yields Node objects, while
        an AStarNode object yields AStarNodes. You can use type(node) to do this.
    """
    def expand_node(self, node: Node) -> Iterator[Node]:
        self.expansion_history.append(node.coord)
        s = node.coord
        for action in self.maze.valid_ordered_action(s):
            new_state = Maze.resulting_coord(node.coord, action)
            new_cost = node.path_cost + self.maze.action_cost(action)
            yield type(node)(new_state, node, action, new_cost)


    """
    Inputs: 
    1. A start node
    2. A frontier (a queue)
    """
    def best_first_search(self, node: Node, frontier: Queue[Node]) -> SearchResult:
        self.clear_expansion_history()

        if self.goal_test(node):
            path, actions = node.trace_back()
            return SearchResult(True, (path, actions), self.expansion_history)  
          
        frontier.put(node)

        reached = {node.coord: node}

        while not frontier.empty():
            current = frontier.get()

            children = self.expand_node(current)

            for child in children:

                if self.goal_test(child):
                    path, actions = child.trace_back()
                    return SearchResult(True, (path, actions), self.expansion_history)

                if child.coord not in reached or child.path_cost < reached[child.coord].path_cost:
                    frontier.put(child)
                    reached[child.coord] = child
                

        return SearchResult(False, None, self.expansion_history)
    
    def bfs(self) -> SearchResult:
        self.clear_expansion_history()
        start = Node(self.maze.start, None, None, 0)
        return self.best_first_search(start, Queue())

    """
    
    NOTE：Although DFS is often implemented as a tree-like search,
    for this it is implemented as a graph search, since we are applying
    it to a state space with cycles that must be handled in some way.
    """
    def dfs(self) -> SearchResult:
        self.clear_expansion_history()
        start = Node(self.maze.start, None, None, 0)
        
        frontier = LifoQueue()
        frontier.put(start)

        reached = {start.coord: start}

        while not frontier.empty():
            current = frontier.get()

            if self.goal_test(current):
                path, actions = current.trace_back()
                return (True, (path, actions), self.expansion_history)
            
            children = self.expand_node(current)
            for child in children:
                if child.coord not in reached:
                    frontier.put(child)
                    reached[child.coord] = child

        return SearchResult(False, None, self.expansion_history)


    def ucs(self) -> SearchResult:
        self.clear_expansion_history()
        start = Node(self.maze.start, None, None, 0)
        frontier = PriorityQueue()
        frontier.put((start.path_cost , start))

        reached = {start.coord: start}

        while not frontier.empty():
            current_cost, current = frontier.get()
            if self.goal_test(current):
                path, actions = current.trace_back()
                return SearchResult(True, (path, actions), self.expansion_history)

            children = self.expand_node(current)
            for child in children:
                s = child.coord

                if s not in reached or child.path_cost < reached[s].path_cost:
                    reached[s] = child  
                    frontier.put((child.path_cost, child))  

        return SearchResult(False, None, self.expansion_history)
    def astar(self) -> SearchResult:
        self.clear_expansion_history()
        
        AStarNode.END_COORDS = self.maze.end
        
        start_node = AStarNode(self.maze.start, None, None, 0)
        
        frontier = PriorityQueue()
        frontier.put(start_node)
        
        reached = {start_node.coord: start_node}

        while not frontier.empty():
            current = frontier.get()  
            
            if self.goal_test(current):
                path, actions = current.trace_back()
                return SearchResult(True, (path, actions), self.expansion_history)

            children = self.expand_node(current)
            for child in children:
                s = child.coord

                if s not in reached or child < reached[s]:
                    reached[s] = child  
                    frontier.put(child)

        return SearchResult(False, None, self.expansion_history)
        
